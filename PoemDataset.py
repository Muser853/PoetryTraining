
    def _load_phonetic_dict(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_dict = json.load(f)

            validated_dict = defaultdict(list)
            for word, phonemes in raw_dict.items():
                clean_word = re.sub(r'\(\d+\)$', '', word).lower().strip()

                if not isinstance(phonemes, list):
                    raise ValueError(f"Invalid phonemes format for {word}, expected list")

                validated_phon = []
                for p in phonemes:
                    if not re.match(r'^[A-Z]+[0-2]?$', p):
                        raise ValueError(f"Invalid phoneme format: {p} in {word}")
                    validated_phon.append(p)

                if not validated_phon:
                    raise ValueError(f"No valid phonemes for {word}")

                validated_dict[clean_word].append(validated_phon)  
                
            final_dict = defaultdict(list)
            for word, variants in validated_dict.items():
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
            raise RuntimeError(f"Failed to load {file_path}: {str(e)}")

    def _get_english_phonemes(self, word):
        word = word.lower().strip()
        variants = self.english_phonetic_dict.get(word, [[]])  # default empty list
        if not variants:
            return []  # return empty list
        first_variant = variants[0]
        return [' '.join(first_variant)] if first_variant else []  # return lsit

    def __getitem__(self, idx):
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
            'structural_tone': structural_tone,
            'structural_rhyme': structural_rhyme,
        }
