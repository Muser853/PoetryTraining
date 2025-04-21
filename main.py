import re
import os
import json
import math
import torch
import torch.nn.functional
from torch import nn
from transformers import BertModel, Trainer, TrainingArguments, BertTokenizer
from bert_score import BERTScorer
from collections import defaultdict

class PoemDataset(torch.utils.data.Dataset):
    def __init__(self, data, max_length=512):
        self.MAX_LINES = 6
        self.data = data
        self.max_length = max_length
        self.chinese_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.english_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.tone_mapping = {
            '平': ['1', '2'], '仄': ['3', '4']
        }  # mapping to flat / sharp tone

        script_dir = os.path.dirname(os.path.abspath(__file__))

        self.chinese_phonetic_dict = self._load_polyphonic_dict(
            os.path.join(
                script_dir,
                'pinyinDict.json'
            )
        )

        self.english_phonetic_dict = self._load_phonetic_dict(
            os.path.join(
                script_dir,
                'PHONETICDICTIONARY/phonetic-dictionary.json'
            )
        )
        '''with open('pinyinDict.json') as f: self.chinese_phonetic_dict = json.load(f)
        with open('PHONETICDICTIONARY/phonetic-dictionary.json') as f: self.english_phonetic_dict = json.load(f)'''
    @staticmethod
    def _load_polyphonic_dict(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_dict = json.load(f)

            validated_dict = defaultdict(list)
            for char, pinyins in raw_dict.items():
                if not isinstance(pinyins, list):
                    raise ValueError(
                        f"Invalid pinyin format for {char}, expected list"
                    )

                # clean the data and preserve polyphones for Chinese characters
                cleaned = [p.strip().lower() for p in pinyins if p.strip()]
                if not cleaned:
                    raise ValueError(
                        f"No valid pinyin for {char}"
                    )

                validated_dict[char] = cleaned

            return validated_dict

        except Exception as e:
            raise RuntimeError(
                f"Failed to load pinyinDict.json: {str(e)}"
            )

    @staticmethod
    def _load_phonetic_dict(file_path):
        try:
            with open(
                    file_path,
                    'r',
                    encoding='utf-8'
            ) as f:
                raw_dict = json.load(f)

            validated_dict = defaultdict(list)

            for word, phonemes in raw_dict.items():
                clean_word = re.sub(
                    r'\(\d+\)$',
                    '',
                    word
                ).lower().strip()

                if not isinstance(phonemes, list):
                    raise ValueError(
                        f"Invalid phonemes format for {word}, expected list"
                    )

                validated_phon = []
                for p in phonemes:
                    if not re.match(r'^[A-Z]+[0-2]?$', p):
                        raise ValueError(f"Invalid phoneme format: {p} in {word}")

                    validated_phon.append(p)

                if not validated_phon:
                    raise ValueError(f"No valid phonemes for {word}")

                # integrate variant pronunciation for the same English word（ZYUGANOV和ZYUGANOV'S）
                validated_dict[clean_word].append(validated_phon)

            # sort multiple pronunciations for the same word（high frequency at the front
            final_dict = defaultdict(list)

            for word, variants in validated_dict.items():
                # simply remove repetitions and sort（can be extended to do better）
                unique_variants = []
                seen = set()

                for v in variants:
                    key = ''.join(v)
                    if key not in seen:
                        seen.add(key)
                        unique_variants.append(v)
                final_dict[word] = unique_variants

            return final_dict

        except Exception as e:
            raise RuntimeError(f"Failed to load phonetic-dictionary.json: {str(e)}")

    @staticmethod
    def _phonetic_similarity(chn_phon, eng_phon):
        # chars matching phonetics
        chn_chars = set(
            chn_phon.replace(' ', '')
        )

        eng_chars = set(
            eng_phon.replace(' ', '')
        )

        common = len(
            chn_chars & eng_chars
        )

        return common / (len(chn_chars) + len(eng_chars))

    @staticmethod
    def check_dataset(dataset):
        invalid_samples = []
        for i in range(len(dataset)):
            sample = dataset[i]
            if sample['input_ids'].sum() == 0: #check if empty
                invalid_samples.append(i)
        print(f"Invalid samples count: {len(invalid_samples)}")

        """
Polysyllabic word selection strategy
So far: The common pronunciation is selected first, and the context-based disambiguation can be extended later
        """

    def _select_pinyin(self, char):
        pinyins = self.chinese_phonetic_dict.get(
            char,
            ['UNK']
        )

        return pinyins[0] if pinyins else 'UNK'

    def _get_english_phonemes(self, word):

        variants = self.english_phonetic_dict.get(word.lower().strip(), [['UNK']])
        if not variants or len(variants[0]) == 0:  # prevent empty list
            return ['UNK']

        return [
            ' '.join(v)
            for v in variants[0]
        ]

    def _get_structural_labels(self, poem):
        labels = []
        valid_poem = [
            line.strip()
            for line in poem
            if len(line.strip()) > 0
        ]

        if not valid_poem:
            return {'tone_labels': [], 'rhyme_scheme': 'A'}

        last_chars = []
        for line in valid_poem:
            if len(line) == 0:
                last_chars.append('*')  # place holder
            else:
                last_chars.append(line[-1])

        current_rhymes = {}
        rhyme_code = []

        for line in poem:
            line_labels = []
            for char in line:
                pinyin = self.chinese_phonetic_dict.get(char, '')
                tone = pinyin[-1] if pinyin else ''
                label = '平' if tone in self.tone_mapping['平'] else '仄'
                line_labels.append(label)

            labels.append(''.join(line_labels))

        for char in last_chars:
            pinyin = self._select_pinyin(char)
            rhyme_part = pinyin[:-1] if pinyin else ''

            if rhyme_part not in current_rhymes:
                current_rhymes[rhyme_part] = chr(ord('A') + len(current_rhymes))

            rhyme_code.append(current_rhymes[rhyme_part])

        return {
            'tone_labels': labels,
            'rhyme_scheme': ''.join(rhyme_code)
        }

    def _align_pronunciation(self, chn_phon, eng_phon):
        alignment_scores = []
        for c_p, e_p in zip(chn_phon, eng_phon):
            score = self._phonetic_similarity(c_p, e_p)
            alignment_scores.append(score)
        return sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0.0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if not (0 < idx <= len(self.data)):
            raise IndexError("Index out of range")

        item = self.data[idx]
        chn_lines = item.get('chinese', [])
        eng_lines = item.get('english', [])

        # non-Empty and line number matches
        if not chn_lines or not eng_lines:
            raise ValueError(f"Invalid data at index {idx}: empty lines")
        if len(chn_lines) != len(eng_lines):
            raise ValueError(f"Mismatched line counts at index {idx}")

        structural_labels = self._get_structural_labels(chn_lines)

        tone_str = ''.join(
            [
                "".join(line)
                for line in structural_labels['tone_labels']
            ]
        )

        rhyme_str = structural_labels['rhyme_scheme']

        structural_tone = torch.tensor(
            [0 if c == '平' else 1 for c in tone_str],
            dtype=torch.long
        )

        structural_rhyme = torch.tensor(
            [ord(c) - ord('A') for c in rhyme_str],
            dtype=torch.long
        )

        src_text = f'[RHYME_SCHEME:{" ".join(
            structural_labels["rhyme_scheme"]
        )}] ' \
                   f'[TONE:{" ".join(
                       ["".join(l) 
                        for l in structural_labels["tone_labels"]
                        ]
                   )
                   }] ' \
                   f'[TEXT] ' + ' '.join(chn_lines)

        chn_phon = []

        for line in chn_lines:
            for c in line:
                pinyins = self.chinese_phonetic_dict.get(c, [''])
                chn_phon.append(pinyins[0] if pinyins else '')

        eng_phon = []

        for word in ' '.join(eng_lines).split():
            phonemes = self._get_english_phonemes(word)
            eng_phon.append(''.join(phonemes) if phonemes else '')

        pron_score = self._align_pronunciation(
            chn_phon,
            eng_phon
        )

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

        pron_features = {
            'half_match': [],
            'consecutive_match': [],
            'odd_even_match': 0.0
        }

        for i in range(len(chn_lines)):
            chn_line = chn_lines[i]
            eng_line = eng_lines[i]

            chn_phons = [
                self.chinese_phonetic_dict.get(
                    c, ['']
                )[0]
                for c in chn_line
            ]

            eng_phons = []

            for w in eng_line.split():
                phonemes = self.english_phonetic_dict.get(
                    w.lower(),
                    [['UNK']]
                )

                eng_phons.append(
                    ''.join(phonemes[0])
                    if phonemes else 'UNK'
                )

            score = self._align_pronunciation(
                chn_phons,
                eng_phons
            )

            pron_features['half_match'].append(score)

        for i in range(len(chn_lines)-1):
            chn1_phon = [
                self.chinese_phonetic_dict.get(
                    c, ['']
                )[0]
                for c in chn_lines[i]
            ]

            chn2_phon = [
                self.chinese_phonetic_dict.get(
                    c, ['']
                )[0]
                for c in chn_lines[i+1]
            ]

            eng1_phon = []
            for w in eng_lines[i].split():
                phonemes = self.english_phonetic_dict.get(
                    w.lower(),
                    [['UNK']]
                )

                eng1_phon.append(
                    ''.join(phonemes[0])
                    if phonemes else 'UNK'
                )

            eng2_phon = []

            for w in eng_lines[i+1].split():
                phonemes = self.english_phonetic_dict.get(
                    w.lower(),
                    [['UNK']]
                )

                eng2_phon.append(
                    ''.join(phonemes[0])
                    if phonemes else 'UNK'
                )

            score = self._align_pronunciation(
                chn1_phon,
                eng1_phon
            ) + self._align_pronunciation(
                chn2_phon,
                eng2_phon
            )

            pron_features[
                'consecutive_match'
            ].append(score)

        odd_phon = ''.join(
            [
                self.chinese_phonetic_dict.get(
                    c, ['']
                )[0]
                for c in ''.join(chn_lines[::2])
            ]
        )

        even_phon = ''.join(
            [
                self.chinese_phonetic_dict.get(
                    c, ['']
                )[0]
                for c in ''.join(chn_lines[1::2])
            ]
        )

        odd_eng_phon = ''.join(
            [
                ''.join(
                    self.english_phonetic_dict.get(
                        w.lower(),
                        [['UNK']]
                    )[0])
                for w in ' '.join(eng_lines[::2]).split()
             ]
        )

        even_eng_phon = ''.join(
            [
                ''.join(
                    self.english_phonetic_dict.get(
                        w.lower(),
                        [['UNK']]
                    )[0])
                for w in ' '.join(eng_lines[1::2]).split()
            ]
        )

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

        half_match = (pron_features['half_match'] + [0] *
            (
                self.MAX_LINES - len(
                    pron_features['half_match']
                )
            )
        )
        consecutive_match = (
            pron_features['consecutive_match'] + [0] *
            (
                    self.MAX_LINES - 1 - len(
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
                consecutive_match[:self.MAX_LINES - 1],
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

# explicit model architecture
class BidirectionalEncoder(nn.Module):  #bidirectional structural awareness encoder
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')

        self.tone_embedding = nn.Embedding(
            2, 768
        )  # flat / sharp embeddings

        self.rhyme_embedding = nn.Embedding(
            10, 768
        )  # rhyming pattern embeddings

        self.fusion_linear = nn.Linear(2304, 768)  # input 768*3 (BERT hidden + tone + rhyme)
        
        # fix BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask, structural_tone, structural_rhyme):
        device = input_ids.device

        input_ids = input_ids.view(
            -1, input_ids.size(-1)
        )

        attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        tone_emb = self.tone_embedding(structural_tone.to(device))
        rhyme_emb = self.rhyme_embedding(structural_rhyme.to(device))

        hidden_mean = hidden_states.mean(dim=1, keepdim=False)
        tone_mean = tone_emb.mean(dim=1, keepdim=False)
        rhyme_mean = rhyme_emb.mean(dim=1, keepdim=False)

        fusion_input = torch.cat(
            [
                hidden_mean,
                tone_mean,
                rhyme_mean
            ], dim=-1
        )

        fusion_gate = torch.sigmoid(self.fusion_linear(fusion_input))
        return hidden_states * fusion_gate.unsqueeze(1)

class PronunciationMatcher(nn.Module):
    def __init__(self, hidden_size=768):
        super().__init__()

        self.alignment_net = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size)
        )

        self.similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, src_phon, tgt_phon):
        # src_phon and tgt_phon are phonetic embedding seq
        alignment = []

        for s, t in zip(src_phon, tgt_phon):
            combined = torch.cat(
                [s, t], dim=-1
            )

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

            x = layer(
                x,
                memory,
                tgt_mask=mask
            )

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

        self.fc = nn.Linear(768, BertTokenizer.from_pretrained('bert-base-uncased').vocab_size)

    def forward(self, input_ids, attention_mask, structural_tone, structural_rhyme,
                labels, half_match, consecutive_match, odd_even_match, rhyme_pattern, pron_score):
        memory = self.encoder(
            input_ids,
            attention_mask,
            structural_tone,
            structural_rhyme
        ).permute(1, 0, 2)  # (seq_len, batch, 768)

        decoder_inputs = labels[ : , : -1]
        decoder_inputs_embed = self.decoder_embedding(decoder_inputs)
        decoder_inputs_embed = decoder_inputs_embed.permute(1, 0, 2)  # (seq_len, batch, 768)

        outputs = self.decoder(
            decoder_inputs_embed, 
            memory,
            rhyme_pattern=rhyme_pattern
        )

        logits = self.fc(
            outputs.permute(1, 0, 2)
        )  # (batch, seq_len, vocab)

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
                  .expand(pronunciation_loss.shape
        ))

        prone_loss = torch.nn.functional.mse_loss(
            pronunciation_loss, target
        )

        return translation_loss + 0.5 * pronunciation_loss + 0.3 * prone_loss

def train():
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
            'rhyme_labels': [0,1,2]
        },
        {
            'title': '静夜思',
            'chinese': ["床前明月光，疑是地上霜。", "举头望明月，低头思故乡"],
            'english': ["Before my bed, the moonlight glows, Like frost upon the ground it flows.", "Gazing at the silvery sphere on high, In its glow, my homeland's memories lie."],
            'rhyme_labels': [0,0]
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
            'rhyme_labels': [0,0,0]
        }
    ])
    print(f"effective training sample amounts: {len([x for x in dataset if x is not None])}")

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
        remove_unused_columns=False
    )

    def compute_metrics(pred):
        # decode prediction and tokens
        pred_text = dataset.english_tokenizer.batch_decode(
            pred.predictions, skip_special_tokens=True
        )

        label_text = dataset.english_tokenizer.batch_decode(
            pred.label_ids, skip_special_tokens=True
        )

        p, r, f1 = BERTScorer(lang="en", rescale_with_baseline=True).score(cands=pred_text,refs=label_text)
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
    gates = [gate.gate_net[0].weight.detach().cpu().numpy() for gate in model.rhythm_gates]
    plt.figure(figsize=(15, 5))

    for idx, gate_weights in enumerate(gates):
        plt.subplot(1, len(gates), idx+1)
        plt.hist(gate_weights.ravel(), bins=30, alpha=0.7, edgecolor='black')
        plt.title(f'Layer {idx} Gate Weights Distribution')
        plt.xlabel('Activation Value')
        plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('rhythm_gates_visualization.png')
    plt.close()

def dynamic_curriculum(current_step, total_steps):
    progress = current_step / total_steps    # Dynamic course learning: Adjust prosody/rhyme loss weights according to training progress
    prosody_weight = 0.8 * (1 + math.cos(math.pi * progress)) / 2 # Adjust prosodic loss weights using cosine annealing (attenuation from 0.8 to 0.2)
    rhyme_weight = 0.2 + 0.8 * progress    # The weight of rhyming loss increases with training
    return prosody_weight, rhyme_weight

if __name__ == '__main__':
    train()