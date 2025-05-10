import os
import json
import math
from typing import Any
import torch
import torch.nn.functional
from torch import nn
from transformers import BertModel, Trainer, TrainingArguments, BertTokenizer
from bert_score import BERTScorer
from collections import defaultdict
from PinYinConverter import PinyinConverter
import re

class PoemDataset(torch.utils.data.Dataset):
    def __init__(self, data, max_length=512):
        self.MAX_LINES = 6
        self.data = data
        self.max_length = max_length
        self.chinese_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.english_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.PinYinConverter = PinyinConverter()

        self.tone_mapping = {
            'flat': ['1', '2', '5'], 'sharp': ['3', '4']
        }  # mapping to flat / sharp tone

        script_dir = os.path.dirname(os.path.abspath(__file__))

        self.chinese_phonetic_dict = self._load_polyphonic_dict(file_path = 'pinyinDict.json')

        self.english_phonetic_dict = self._load_phonetic_dict(
            os.path.join(
                script_dir,
                'PHONETICDICTIONARY/English-phonetic-transcription.json'
            )
        )
        # with open('pinyinDict.json') as f: self.chinese_phonetic_dict = json.load(f)
        # with open('PHONETICDICTIONARY/phonetic-dictionary.json') as f: self.english_phonetic_dict = json.load(f)
    def _load_polyphonic_dict(self, file_path) -> defaultdict[Any, list]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_dict = json.load(f)

            validated_dict = defaultdict(list)

            for char, pinyins in raw_dict.items():
                cleaned = [
                    self.PinYinConverter.reverse_convert(p.strip().lower())
                    for p in pinyins
                    if p.strip()
                ]
                if not cleaned:
                    raise ValueError(f"No valid pinyin for {char}")

                validated_dict[char] = cleaned

            return validated_dict

        except Exception as e:
            raise RuntimeError(
                f"Failed to load pinyinDict.json: {str(e)}"
            )

    @staticmethod
    def _load_phonetic_dict(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_dict = json.load(f)

            validated_dict = defaultdict(list)
            for word, phonemes in raw_dict.items():
                clean_word = re.sub(r'^"|"$', '', word).lower().strip()  # remove ""

                if isinstance(phonemes, str):
                    variants = [phonemes.split()]
                elif isinstance(phonemes, list):
                    variants = []
                    for p in phonemes:
                        if isinstance(p, str):
                            variants.extend(p.split())
                        elif isinstance(p, list):
                            variants.extend(p)
                        else:
                            raise ValueError(f"Invalid phonemes format for {word}")
                else:
                    raise ValueError(f"Invalid phonemes format for {word}")

                validated_phon = []

                for variant in variants:
                    variant_lower = [
                        v.lower()
                        for v in variant
                    ]

                    validated_phon.append(' '.join(variant_lower))

                validated_dict[clean_word] = validated_phon

            return validated_dict
        except Exception as e:
            raise RuntimeError(
                f"Failed to load {file_path}: {str(e)}"
            )

    @staticmethod
    def _phonetic_similarity(chn_phon, eng_phon):
        chn_chars = set(chn_phon.replace(' ', ''))
        eng_chars = set(eng_phon.replace(' ', ''))
        common = len(chn_chars & eng_chars)
        return common / (len(chn_chars) + len(eng_chars))

    def _select_pinyin(self, char):
        """
Polysyllabic word selection strategy
So far: The common pronunciation is selected first, and the context-based disambiguation can be extended later
        """
        pinyins = self.chinese_phonetic_dict.get(char, [])
        return pinyins[0] if pinyins else ''  # return the first so far

    def _get_english_phonemes(self, word):
        word = word.lower().strip()
        variants = self.english_phonetic_dict.get(word, [])

        if variants:

            if isinstance(variants[0], list):
                return [' '.join(v) for v in variants[0]]
            else:
                return [' '.join(variants)]
        return []

    @staticmethod
    def _extract_tone_from_pinyin(pinyin_str):
        tone_map = {
            'ā': '1', 'á': '2', 'ǎ': '3', 'à': '4',
            'ē': '1', 'é': '2', 'ě': '3', 'è': '4',
            'ī': '1', 'í': '2', 'ǐ': '3', 'ì': '4',
            'ō': '1', 'ó': '2', 'ǒ': '3', 'ò': '4',
            'ū': '1', 'ú': '2', 'ǔ': '3', 'ù': '4',
            'ǖ': '1', 'ǘ': '2', 'ǚ': '3', 'ǜ': '4'
        }
        for c in pinyin_str:
            if c in tone_map:
                return tone_map[c]
        return '5'

    def _get_structural_labels(self, poem):
        valid_poem = [
            line.strip()
            for line in poem
            if len(line.strip()) > 0
        ]

        last_chars = []

        for line in valid_poem:
            if not line:
                last_chars.append('*')
            else:
                last_chars.append(line[-1])

        labels = []
        current_rhymes = {}
        rhyme_code = []

        for line in valid_poem:
            line_labels = []

            for char in line:
                pinyin = self._select_pinyin(char)
                tone = self._extract_tone_from_pinyin(pinyin)
                label = 'flat' if tone in ['1', '2', '5'] else 'sharp'
                line_labels.append(label)

            labels.append(''.join(line_labels))

        for char in last_chars:
            pinyin = self._select_pinyin(char)
            rhyme_part = pinyin[:-1] if pinyin else ''

            if rhyme_part not in current_rhymes: current_rhymes[rhyme_part] = chr(ord('A') + len(current_rhymes))

            rhyme_code.append(current_rhymes[rhyme_part])

        return {
            'tone_labels': labels,
            'rhyme_scheme': ''.join(rhyme_code)
        }

    def _align_pronunciation(self, chn_phon, eng_phon):
        alignment_scores = []
        for c_p, e_p in zip(chn_phon, eng_phon):
            e_p_clean = e_p.lower().strip()

            e_pinyin = self.english_phonetic_dict.get(e_p_clean, [['UNK']])
            if e_pinyin and isinstance(e_pinyin[0], list) and e_pinyin[0]:
                e_pinyin = e_pinyin[0][0]
            else:
                e_pinyin = 'UNK'

            score = self._phonetic_similarity(c_p, e_pinyin)
            alignment_scores.append(score)

        return sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0.0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            item = self.data[idx]
            chn_lines = item['chinese']
            structural_features = self._get_structural_labels(chn_lines)

            structural_tone = torch.tensor(
                [0 if c == 'flat' else 1 for c in ''.join(structural_features['tone_labels'])],
                dtype=torch.long
            )
            structural_rhyme = torch.tensor(
                [ord(c) - ord('A') for c in structural_features['rhyme_scheme']],
                dtype=torch.long
            )
            # structural labeling
            src_text = f'[RHYME_SCHEME:{" ".join(structural_features["rhyme_scheme"])}] ' \
                f'[TONE:{" ".join(["".join(l) for l in structural_features["tone_labels"]])}] ' \
                f'[TEXT] ' + ' '.join(chn_lines)

            eng_lines = item['english']

            src_encoding = self.chinese_tokenizer(
                src_text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            eng_encoding = self.english_tokenizer(
                ' '.join(eng_lines),
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            # feature extraction
            chn_phon = [self.chinese_phonetic_dict.get(c, [''])[0] for c in ''.join(chn_lines)]
            eng_phon = []

            for w in ' '.join(eng_lines).split():
                phonemes = self._get_english_phonemes(w)

                eng_phon.append(
                    ' '.join(phonemes)
                    if phonemes else ''
                )

            pron_score = self._align_pronunciation(chn_phon, eng_phon)
            pron_features = {  # feature calculation
                'half_match': [],
                'consecutive_match': [],
                'odd_even_match': 0.0
            }
            for chn_line, eng_line in zip(chn_lines, eng_lines):
                # pinyin extraction with 1 character having multiple pronunciations
                chn_phons = [
                    self.chinese_phonetic_dict.get(
                        c, ['']
                    )[0]
                    for c in chn_line
                ]

                eng_phons = []

                for w in eng_line.split():
                    phonemes = self._get_english_phonemes(w)

                    eng_phons.append(
                        ' '.join(phonemes)
                        if phonemes else ''
                    )

                score = self._align_pronunciation(chn_phons, eng_phons)
                pron_features['half_match'].append(score)

            for i in range(len(chn_lines) - 1):
                chn1_phon = [
                    self.chinese_phonetic_dict.get(c, [''])[0]
                    for c in chn_lines[i]
                ]

                chn2_phon = [
                    self.chinese_phonetic_dict.get(c, [''])[0]
                    for c in chn_lines[i + 1]
                ]

                eng1_phon = [
                    ' '.join(self.english_phonetic_dict.get(w.lower(), [''])[0])
                    for w in eng_lines[i].split()
                ]
                eng2_phon = [
                    ' '.join(self.english_phonetic_dict.get(w.lower(), [''])[0])
                    for w in eng_lines[i + 1].split()
                ]

                score = self._align_pronunciation(chn1_phon, eng1_phon) + self._align_pronunciation(chn2_phon, eng2_phon)
                pron_features['consecutive_match'].append(score)

            odd_phon = ''.join(
                [
                    p for line in chn_lines[::2]
                    for c in line
                    for p in self.chinese_phonetic_dict.get(c, [''])
                ]
            )

            even_phon = ''.join(
                [
                    p for line in chn_lines[1::2]
                    for c in line
                    for p in self.chinese_phonetic_dict.get(c, [''])
                ]
            )

            odd_eng_phon = ''.join(
                [
                ' '.join(p)
                    for line in eng_lines[::2]
                    for w in line.split()
                    for p in self.english_phonetic_dict.get(w.lower(), [''])
                ]
            )
            even_eng_phon = ''.join([
                ' '.join(p)
                for line in eng_lines[1::2]
                for w in line.split()
                for p in self.english_phonetic_dict.get(w.lower(), [''])
            ])

            pron_features['odd_even_match'] = (
                self._align_pronunciation(
                    odd_phon,
                    odd_eng_phon
                ) +
                self._align_pronunciation(
                    even_phon,
                    even_eng_phon
                )
            ) / 2.0

            half_match = (pron_features['half_match'] + [0]*
                (
                    self.MAX_LINES - len(
                        pron_features['half_match']
                    )
                )
            )
            consecutive_match = (
                pron_features['consecutive_match'] + [0]*
                (
                        self.MAX_LINES-1 - len(
                             pron_features['consecutive_match']
                      )
                )
            )
            
            return {
                'input_ids': src_encoding[
                    'input_ids'
                ].squeeze(0),

                'attention_mask': src_encoding[
                    'attention_mask'
                ].squeeze(0),

                'labels': eng_encoding[
                    'input_ids'
                ].squeeze(0),

                "half_match": torch.tensor(
                    half_match[:self.MAX_LINES],
                    dtype=torch.float
                ).unsqueeze(0),

                'consecutive_match': torch.tensor(
                    consecutive_match[:self.MAX_LINES-1],
                    dtype=torch.float
                ).unsqueeze(0),

                'structural_tone': structural_tone,
                'structural_rhyme': structural_rhyme,
                'rhyme_pattern': structural_rhyme,

                "odd_even_match": torch.tensor(
                    pron_features['odd_even_match']
                ).unsqueeze(0),

                "pron_score": torch.tensor(pron_score).unsqueeze(0),
            }
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            return None # index out of range

# explicit model architecture
class BidirectionalEncoder(nn.Module):  # bidirectional structural awareness encoder
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')

        self.tone_embedding = nn.Embedding(
            2,
            768
        )  # flat / sharp embeddings

        self.rhyme_embedding = nn.Embedding(
            10,
            768
        )  # rhyming pattern embeddings

        self.fusion_linear = nn.Linear(2304,768)  # input 768*3 (BERT hidden + tone + rhyme)

        # fix BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, input_ids,
                attention_mask,
                structural_tone,
                structural_rhyme):

        device = input_ids.device
        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask
        )
        hidden_states = outputs.last_hidden_state
        tone_emb = self.tone_embedding(structural_tone.to(device))
        rhyme_emb = self.rhyme_embedding(structural_rhyme.to(device))

        fusion_input = torch.cat([
            hidden_states.mean(dim=1),
            tone_emb.mean(dim=1),
            rhyme_emb.mean(dim=1)
            ], dim=-1
        )

        fusion_gate = torch.sigmoid(self.fusion_linear(fusion_input))
        return hidden_states * fusion_gate.unsqueeze(1)

class PronunciationMatcher(nn.Module):
    def __init__(self, hidden_size=768):
        super().__init__()
        self.alignment_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size)
        )
        self.similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, src_phon, tgt_phon):
        # src_phon and tgt_phon are phonetic embedding seq
        alignment = []

        for s, t in zip(src_phon, tgt_phon):
            combined = torch.cat([s, t], dim=-1)
            aligned = self.alignment_net(combined)
            alignment.append(self.similarity(aligned, s))

        return torch.stack(alignment).mean()

class StructureAwareDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=768, nhead=8)
            for _ in range(6)
        ])

        self.pronunciation_match = PronunciationMatcher()

    @staticmethod
    def _generate_rhyme_mask(pattern, seq_len, device):
        mask = torch.ones(seq_len, seq_len, device=device)

        for pos in pattern:
            mask[pos, :] *= 2.0

        return mask

    def forward(self, x, memory, rhyme_pattern):
        for layer in self.layers:
            mask = self._generate_rhyme_mask(
                pattern=rhyme_pattern,
                seq_len=x.size(0),
                device=x.device
            )

            x = layer(x, memory, tgt_mask=mask)

        pron_scores = self.pronunciation_match(memory, x)
        return x + pron_scores.unsqueeze(-1)

class PoetryTranslator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = BidirectionalEncoder()
        self.decoder = StructureAwareDecoder()
        self.pronunciation_loss = nn.CosineEmbeddingLoss()

        self.decoder_embedding = nn.Embedding(
            BertTokenizer.from_pretrained('bert-base-uncased').vocab_size, 
            768
        )

        self.fc = nn.Linear(
            768, BertTokenizer.from_pretrained('bert-base-uncased').vocab_size
        )

    def forward(self,
                input_ids,
                attention_mask,
                structural_tone,
                structural_rhyme,
                labels, half_match,
                consecutive_match,
                odd_even_match,
                rhyme_pattern,
                pron_score):

        memory = self.encoder(
            input_ids,
            attention_mask,
            structural_tone,
            structural_rhyme

        ).permute(1, 0, 2)# (seq_len, batch, 768)

        decoder_inputs = labels[ : , : -1]

        decoder_emb = self.decoder_embedding(
            decoder_inputs
        ).permute(1, 0, 2)

        outputs = self.decoder(
            decoder_emb, memory, rhyme_pattern
        )

        logits = self.fc(
            outputs.permute(1, 0, 2)
        )# (batch, seq_len, vocab)

        translation_loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels[:, 1:].contiguous().view(-1)
        )

        pronunciation_loss = (
                half_match.mean() +
                consecutive_match.mean() +
                odd_even_match +
                pron_score.mean()
        )

        target = (torch.tensor(
            0.8, device=input_ids.device)
                  .expand_as(pronunciation_loss
            )
        )

        return translation_loss + 0.5 * pronunciation_loss + 0.3 * torch.nn.functional.mse_loss(pronunciation_loss, target)


# constraint generation module
class RhymeConstrainedGenerator:
    def __init__(self, model, beam_scorer, english_phonetic_dict):
        self.model = model
        self.beam_scorer = beam_scorer
        self.english_phonetic_dict = english_phonetic_dict()

    def generate(self, inputs):
        # dynamically add rhyming constraints
        constraints = []
        for line_idx, rhyme_char in enumerate(inputs['rhyme_pattern']):
            target_phonemes = self.english_phonetic_dict.get(
                rhyme_char, ['']
            )[0]

            constraints.append({
                'position': -1,
                'phonemes': [target_phonemes]
            })

        self.beam_scorer.constraints = constraints

        return self.model.generate(
            inputs,
            beam_scorer=self.beam_scorer,
            max_length=128
        )

def train():  # training and evaluation
    dataset = PoemDataset([
        {
            'title': '静女',
            'chinese': [
                "静女其姝，俟我于城隅。爱而不见，搔首踟蹰。",
                "静女其娈，贻我彤管。彤管有炜，说怿女美。",
                "自牧归荑，洵美且异。匪女之为美，美人之贻。"
            ],
            'english': [
                "The maiden is fair, Awaiting me around the corner. Loving me, she can’t find me, Fidgeting while wondering.",
                "The maiden is cute, Vouchsafing to me a vermillion flute. The flute has luster, Mirroring her bright beauty.",
                "I brought her grass Tee from field, Fairly fair and far. I love what the nature offers, When it’s presented to her."
            ],
            'rhyme_labels': [0, 1, 2]
        },
        {
            'title': '静夜思',
            'chinese': ["床前明月光，疑是地上霜。", "举头望明月，低头思故乡"],
            'english': ["Before my bed, the moonlight glows, Like frost upon the ground it flows.",
                        "Gazing at the silvery sphere on high, In its glow, my homeland's memories lie."],
            'rhyme_labels': [0, 0]
        },
        {
            'title': '泂酌彼行潦',
            'chinese': [
                "泂酌彼行潦，挹彼注兹，可以餴饎。岂弟君子，民之父母。",
                "泂酌彼行潦，挹彼注兹，可以濯罍。岂弟君子，民之攸归。",
                "泂酌彼行潦，挹彼注兹，可以濯溉。岂弟君子，民之攸塈。"
            ],
            'english': [
                "Fetch water from afar through the wayside seeper, Scoop out from that pan and spill into this can, With which he can steam rice. What an amiable gentleman, By whom the people are reared.",
                "Fetch water from afar through the wayside seeper, Scoop out from that pan and spill into this can, With which he can water stein. What an amiable gentleman, To whom the people belong.",
                "Fetch water from afar through the wayside seeper, Scoop out from that pan and spill into this can, With which he can wash chalice. What an amiable gentleman, For whom the people yearn."
            ],
            'rhyme_labels': [0, 0, 0]
        }
    ])

    model = PoetryTranslator()

    args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=16,
        max_steps=512,
        learning_rate=3e-5,
        per_device_train_batch_size=1,
        logging_steps=128,
        save_strategy='steps',
        fp16=False,
        use_cpu=True,
        remove_unused_columns = False
    )

    def compute_metrics(pred):
        # decode prediction and tokens
        pred_text = [
            dataset.english_tokenizer.Decode(ids)
            for ids in pred.predictions
        ]

        label_text = [
            dataset.english_tokenizer.Decode(ids)
            for ids in pred.label_ids
        ]

        p, r, f1 = BERTScorer(lang="en", rescale_with_baseline=True).score(cands=pred_text, refs=label_text)

        rhyme_acc = 0.0  # calculate rhyming accuracy
        for pred_line, label_line in zip(pred_text, label_text):
            # predict rhyming
            pred_rhyme = pred_line.split()[-1][-1].lower()
            label_rhyme = label_line.split()[-1][-1].lower()
            rhyme_acc += 1 if pred_rhyme == label_rhyme else 0

        return {
            'bert_score_precision': p.mean().item(),
            'bert_score_recall': r.mean().item(),
            'bert_score_f1': f1.mean().item(),
            'rhyme_acc': rhyme_acc / len(pred_text)
        }

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        compute_metrics=compute_metrics
    )
    trainer.train()

"""
1. Structure enhancement encoder: 
- Bidirectional BERT encoder incorporates tonal and rhyming embeddings 
- The gating mechanism dynamically incorporates semantic and structural features 
- Freeze pre-training parameters + fine-tune top layer 

2. Pronunciation consistency processing:
- Phoneme similarity calculation module 
- Multi-task learning framework joint optimization 

3. Constraint generation: 
- Increased focus on rhyming positions 
- Dynamic rhythm mask mechanism 

4. Evaluation system: 
- BERTScore Indicates semantic similarity 
- Custom rhyming accuracy metrics 
- Sound alignment visualization tool
"""
# helper methods
def visualize_rhythm_gates(model):
    """activation values"""
    import matplotlib.pyplot as plt

    gates = [
        gate.gate_net[0].weight.detach().cpu().numpy()
        for gate in model.rhythm_gates
    ]

    plt.figure(figsize=(15, 5))

    for idx, gate_weights in enumerate(gates):
        plt.subplot(1, len(gates), idx + 1)

        plt.hist(
            gate_weights.ravel(),
            bins=30,
            alpha=0.7,
            edgecolor='black'
        )

        plt.title(
            f'Layer {idx} Gate Weights Distribution'
        )

        plt.xlabel(
            'Activation Value'
        )

        plt.ylabel(
            'Frequency'
        )

    plt.tight_layout()
    plt.savefig('rhythm_gates_visualization.png')
    plt.close()

def dynamic_curriculum(current_step, total_steps):
    progress = current_step / total_steps
    # Dynamic course learning: Adjust prosody/rhyme loss weights according to training progress
    prosody_weight = 0.8 * (1 + math.cos(
        math.pi * progress)
    ) / 2  # Adjust prosodic loss weights using cosine annealing (attenuation from 0.8 to 0.2)

    rhyme_weight = 0.2 + 0.8 * progress  # The weight of rhyming loss increases with training
    return prosody_weight, rhyme_weight

if __name__ == '__main__':
    train()