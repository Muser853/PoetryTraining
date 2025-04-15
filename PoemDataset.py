
    def _load_phonetic_dict(self, file_path):
        # 修正发音词典加载逻辑，与main.py保持一致
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_dict = json.load(f)

            validated_dict = defaultdict(list)
            for word, phonemes in raw_dict.items():
                clean_word = re.sub(r'\(\d+\)$', '', word).lower().strip()  # 使用与main.py相同的正则清洗

                if not isinstance(phonemes, list):
                    raise ValueError(f"Invalid phonemes format for {word}, expected list")

                validated_phon = []
                for p in phonemes:
                    if not re.match(r'^[A-Z]+[0-2]?$', p):  # 保持与main.py一致的正则验证
                        raise ValueError(f"Invalid phoneme format: {p} in {word}")
                    validated_phon.append(p)

                if not validated_phon:
                    raise ValueError(f"No valid phonemes for {word}")

                # 修正存储结构为列表嵌套列表
                validated_dict[clean_word].append(validated_phon)  

            # 处理变体去重
            final_dict = defaultdict(list)
            for word, variants in validated_dict.items():
                unique_variants = []
                seen = set()
                for v in variants:
                    key = ''.join(v)
                    if key not in seen:
                        seen.add(key)
                        unique_variants.append(v)
                final_dict[word] = unique_variants  # 保持与main.py一致的最终结构

            return final_dict

        except Exception as e:
            raise RuntimeError(f"Failed to load {file_path}: {str(e)}")

    def _get_english_phonemes(self, word):
        # 修正发音获取逻辑，返回第一个变体的发音列表
        word = word.lower().strip()
        variants = self.english_phonetic_dict.get(word, [[]])  # 确保默认值是空列表
        if not variants:
            return []  # 返回空列表而非空字符串
        first_variant = variants[0]
        return [' '.join(first_variant)] if first_variant else []  # 返回列表形式

    def __getitem__(self, idx):
        # 修正结构特征处理方式，与main.py保持一致
        structural_labels = self._get_structural_labels(chn_lines)
        structural_tone = torch.tensor(
            [0 if c == '平' else 1 for c in ''.join(structural_labels['tone_labels'])],
            dtype=torch.long
        )
        structural_rhyme = torch.tensor(
            [ord(c) - ord('A') for c in structural_labels['rhyme_scheme']],
            dtype=torch.long
        )
        return {
            'structural_tone': structural_tone,  # 显式传递结构特征
            'structural_rhyme': structural_rhyme,
        }