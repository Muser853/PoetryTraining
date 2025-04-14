import torch
import json
import matplotlib.pyplot as plt
from bert_score import BERTScorer
from transformers import BertTokenizer
from typing import List, Dict
from main import PoetryTranslator


class PoetryTranslationSystem:
    def __init__(self, model_path: str, device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._init_model(model_path)
        self.scorer = BERTScorer(lang="en", rescale_with_baseline=True)
        self.tokenizers = self._init_tokenizers()
        self.phonetic_dicts = self._load_phonetic_dicts()

    def _init_model(self, model_path: str) -> PoetryTranslator:
        model = PoetryTranslator()
        try:
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        except RuntimeError as e:
            raise RuntimeError(f"model loading failed: {str(e)}. Please check if the model framework fits") from e
        model.eval()
        return model.to(self.device)

    @staticmethod
    def _init_tokenizers() -> Dict[str, BertTokenizer]:
        return {
            'chinese': BertTokenizer.from_pretrained("bert-base-chinese"), # word splitter
            'english': BertTokenizer.from_pretrained("bert-base-uncased")
        }

    @staticmethod
    def _load_phonetic_dicts() -> Dict[str, dict]:
        try:
            with open("pinyinDict.json") as f:
                chn_phonetic = json.load(f)
            with open("PHONETICDICTIONARY/phonetic-dictionary.json") as f:
                eng_phonetic = json.load(f)
            return {'chinese': chn_phonetic, 'english': eng_phonetic}
        except FileNotFoundError as e:
            raise FileNotFoundError(f"no photic dictionary found: {str(e)}") from e

    def _calculate_rhythm_match(self, src_poem: str, translation: str) -> float:
        src_lines = [line.strip() for line in src_poem.split('，') if line.strip()]
        tgt_lines = [line.strip() for line in translation.split('\n') if line.strip()]

        if not src_lines or not tgt_lines:
            return 0.0

        src_rhythm = []
        for char in ''.join(src_lines):
            pinyins = self.phonetic_dicts['chinese'].get(char, [''])
            src_rhythm.append('平' if pinyins and pinyins[0][-1] in {'1', '2'} else '仄')

        # end rhyme
        tgt_rhythm = []
        for line in tgt_lines:
            words = line.split()
            if not words:
                continue
            last_word = words[-1].lower()
            stress = self.phonetic_dicts['english'].get(last_word, [''])[0]
            tgt_rhythm.append('强' if '1' in stress else '弱' if '0' in stress else '')

        # 简单比例匹配（实际可改进为动态规划对齐）
        match_count = sum(1 for s, t in zip(src_rhythm, tgt_rhythm) if s and t and s[0] == t[0])
        return match_count / max(len(src_rhythm), 1)

    def _calculate_rhyme_accuracy(self, translation: str) -> float:

        lines = [line for line in translation.split('\n') if line.strip()]
        if len(lines) < 2:
            return 0.0

        last_words = [line.strip().split()[-1].lower() for line in lines]
        phonetics = []
        for word in last_words:
            phons = self.phonetic_dicts['english'].get(word, [''])
            phonetics.append(phons[0] if phons else '')

        base_phon = phonetics[0]
        matches = sum(1 for p in phonetics[1:] if p and p[-3:] == base_phon[-3:])
        return matches / (len(phonetics) - 1)

    def _analyze_poem_structure(self, poem: str) -> Dict:
        lines = [line.strip() for line in poem.split('，') if line.strip()]
        if not lines:
            raise ValueError("wrong format")

        # flat or sharp tone determination
        tone_labels = []
        for line in lines:
            line_tones = []
            for char in line:
                pinyins = self.phonetic_dicts['chinese'].get(char, [''])
                tone = pinyins[0][-1] if pinyins else ''
                label = '平' if tone in {'1', '2'} else '仄' if tone else '未知'
                line_tones.append(label)
            tone_labels.append(''.join(line_tones))

        # rhyming pattern analysis
        rhyme_scheme = []
        rhyme_map = {}
        for line in lines:
            last_char = line[-1]
            pinyins = self.phonetic_dicts['chinese'].get(last_char, [''])
            rhyme_part = pinyins[0][:-1] if pinyins else ''

            if rhyme_part not in rhyme_map:
                rhyme_map[rhyme_part] = chr(65 + len(rhyme_map))
            rhyme_scheme.append(rhyme_map[rhyme_part])

        return {
            'tone_matrix': tone_labels,
            'rhyme_scheme': ''.join(rhyme_scheme),
            'line_count': len(lines)
        }

    def visualize_prosody(self, poem: str, save_path: str = None) -> None:
        structure = self._analyze_poem_structure(poem)
        plt.figure(figsize=(15, 8))

        # flat or sharp matrix visualization
        plt.subplot(2, 2, 1)
        tones = [t for line in structure['tone_matrix'] for t in line]
        plt.hist(tones, bins=3, edgecolor='black')
        plt.title('Tone Distribution')
        plt.xlabel('Tone Category')
        plt.ylabel('Frequency')

        # rhyming pattern
        plt.subplot(2, 2, 2)
        rhyme_labels = list(structure['rhyme_scheme'])
        plt.bar(range(len(rhyme_labels)), [ord(c) - 64 for c in rhyme_labels])
        plt.xticks(range(len(rhyme_labels)), rhyme_labels)
        plt.title('Rhyme Scheme Pattern')

        # the structures of lines
        plt.subplot(2, 1, 2)
        line_lengths = [len(line) for line in poem.split('，')]
        plt.plot(line_lengths, marker='o')
        plt.title('Line Structure Analysis')
        plt.xlabel('Line Number')
        plt.ylabel('Characters per Line')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()

    def batch_translate(self, poems: List[str], batch_size: int = 4) -> List[str]:
        translations = []
        for i in range(0, len(poems), batch_size):
            batch = poems[i:i + batch_size]
            inputs = self.tokenizers['chinese'](
                [self._format_input(p) for p in batch],
                return_tensors='pt',
                padding='longest',
                truncation=True,
                max_length=512
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=int(2.5 * max(len(p) for p in batch)),
                    num_beams=5,
                    early_stopping=True
                )

            translations.extend([
                self._postprocess(self.tokenizers['english'].decode(ids, skip_special_tokens=True))
                for ids in outputs
            ])
        return translations

    def translate(self, poem: str) -> str:
        inputs = self.tokenizers['chinese'](
            self._format_input(poem),
            return_tensors='pt',
            padding='longest',
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=int(2.5 * len(poem)),
                num_beams=5,
                early_stopping=True
            )

        return self._postprocess(self.tokenizers['english'].decode(outputs[0], skip_special_tokens=True))


    def evaluate_translation(self, src_poem: str, translation: str) -> Dict:
        P, R, F1 = self.scorer.score([translation], [src_poem])

        rhyme_acc = self._calculate_rhyme_accuracy(translation)
        rhythm_score = self._calculate_rhythm_match(src_poem, translation)

        return {
            'bert_score': F1.mean().item(),
            'rhyme_accuracy': rhyme_acc,
            'rhythm_similarity': rhythm_score,
            'length_ratio': len(translation) / len(src_poem)
        }

    def interactive_session(self) -> None:
        print("===  ===")
        print("available commands:\n:q quit\n:v visual\n:e eval\n:help")
        while True:
            try:
                user_input = input("\nInput poems / commands: ").strip()
                if not user_input:
                    continue

                if user_input.startswith(':q'):
                    break
                elif user_input.startswith(':v'):
                    poem = user_input[2:].strip()
                    self.visualize_prosody(poem)
                elif user_input.startswith(':e'):
                    poem = user_input[2:].strip()
                    translation = self.translate(poem)
                    metrics = self.evaluate_translation(poem, translation)
                    print(f"\nEvaluation Results:\n{self._format_metrics(metrics)}")
                elif user_input.startswith(':help'):
                    self._show_help()
                else:
                    translation = self.translate(user_input)
                    print(f"\nTranslation Results:\n{translation}")

            except Exception as e:
                print(f"process error: {str(e)}")

    def _format_input(self, poem: str) -> str:
        structure = self._analyze_poem_structure(poem)
        return (
            f"[RHYME_SCHEME:{' '.join(structure['rhyme_scheme'])}] "
            f"[TONE:{' '.join(structure['tone_matrix'])}] "
            f"[TEXT]{poem}"
        )

    @staticmethod
    def _postprocess(text: str) -> str:

        lines = []
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
            if line[-1] not in {'.', ',', '!', '?'}:
                line += '.'
            lines.append(line[0].upper() + line[1:])
        return '\n'.join(lines)

    @staticmethod
    def _format_metrics(metrics: Dict) -> str:
        """格式化评估结果"""
        return (
            f"• BERT分数: {metrics['bert_score']:.3f}\n"
            f"• 押韵准确率: {metrics['rhyme_accuracy']:.1%}\n"
            f"• 节奏相似度: {metrics['rhythm_similarity']:.2f}\n"
            f"• 长度比例: {metrics['length_ratio']:.2f}"
        )

    @staticmethod
    def _show_help() -> None:
        """显示帮助信息"""
        print("""
        系统命令说明:
        :q      - 退出系统
        :v 诗歌 - 可视化诗歌韵律结构
        :e 诗歌 - 评估翻译质量
        :help   - 显示本帮助

        示例输入:
        床前明月光，疑是地上霜。
        :v 春风得意马蹄疾，一日看尽长安花
        """)


if __name__ == "__main__":
    try:
        translator = PoetryTranslationSystem(
            model_path="/Users/wangmin/Desktop/TransPOE/RHYTHM/poetry_translator_20250412_180917.pth"
        )
        translator.interactive_session()

    except FileNotFoundError as e:
        print(f"关键文件缺失: {str(e)}")
    except RuntimeError as e:
        print(f"模型初始化失败: {str(e)}")
    except Exception as e:
        print(f"系统初始化异常: {str(e)}")