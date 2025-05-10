import re
from typing import Dict

class PinyinConverter:
    TONE_MAP = {
        'a': ['ā', 'á', 'ǎ', 'à'], 'e': ['ē', 'é', 'ě', 'è'],
        'i': ['ī', 'í', 'ǐ', 'ì'], 'o': ['ō', 'ó', 'ǒ', 'ò'],
        'u': ['ū', 'ú', 'ǔ', 'ù'], 'ü': ['ǖ', 'ǘ', 'ǚ', 'ǜ']
    }
    REVERSE_TONE_MAP = {
        char: (base, str(i + 1))
        for base, tones in TONE_MAP.items()
        for i, char in enumerate(tones)
    }
    PINYIN_PATTERN = r'(shuang|chuang|zhuang|xian|qiong|shuai|niang|guang|sheng|kuang|shang|jiong|huang|jiang|shuan|xiong|zhang|zheng|zhong|zhuai|zhuan|qiang|chang|liang|chuan|cheng|chong|chuai|hang|peng|chuo|piao|pian|chua|ping|yang|pang|chui|chun|chen|chan|chou|chao|chai|zhun|mang|meng|weng|shai|shei|miao|zhui|mian|yong|ming|wang|zhuo|zhua|shao|yuan|bing|zhen|fang|feng|zhan|zhou|zhao|zhei|zhai|rang|suan|reng|song|seng|dang|deng|dong|xuan|sang|rong|duan|cuan|cong|ceng|cang|diao|ruan|dian|ding|shou|xing|zuan|jiao|zong|zeng|zang|jian|tang|teng|tong|bian|biao|shan|tuan|huan|xian|huai|tiao|tian|hong|xiao|heng|ying|jing|shen|beng|kuan|kuai|nang|neng|nong|juan|kong|nuan|keng|kang|shua|niao|guan|nian|ting|shuo|guai|ning|quan|qiao|shui|gong|geng|gang|qian|bang|lang|leng|long|qing|ling|luan|shun|lian|liao|zhi|lia|liu|qin|lun|lin|luo|lan|lou|qiu|gai|gei|gao|gou|gan|gen|lao|lei|lai|que|gua|guo|nin|gui|niu|nie|gun|qie|qia|jun|kai|kei|kao|kou|kan|ken|qun|nun|nuo|xia|kua|kuo|nen|kui|nan|nou|kun|jue|nao|nei|hai|hei|hao|hou|han|hen|nai|rou|xiu|jin|hua|huo|tie|hui|tun|tui|hun|tuo|tan|jiu|zai|zei|zao|zou|zan|zen|eng|tou|tao|tei|tai|zuo|zui|xin|zun|jie|jia|run|diu|cai|cao|cou|can|cen|die|dia|xue|rui|cuo|cui|dun|cun|cin|ruo|rua|dui|sai|sao|sou|san|sen|duo|den|dan|dou|suo|sui|dao|sun|dei|zha|zhe|dai|xun|ang|ong|wai|fen|fan|fou|fei|zhu|wei|wan|min|miu|mie|wen|men|lie|chi|cha|che|man|mou|mao|mei|mai|yao|you|yan|chu|pin|pie|yin|pen|pan|pou|pao|shi|sha|she|pei|pai|yue|bin|bie|yun|nüe|lve|shu|ben|ban|bao|bei|bai|lüe|nve|ren|ran|rao|xie|re|ri|si|su|se|ru|sa|cu|ce|ca|ji|ci|zi|zu|ze|za|hu|he|ha|ju|ku|ke|qi|ka|gu|ge|ga|li|lu|le|qu|la|ni|xi|nu|ne|na|ti|tu|te|ta|xu|di|du|de|bo|lv|ba|ai|ei|ao|ou|an|en|er|da|wu|wa|wo|fu|fo|fa|nv|mi|mu|yi|ya|ye|me|mo|ma|pi|pu|po|yu|pa|bi|nü|bu|lü|e|o|a)'

    def __init__(self):
        self._build_accent_maps()
        self._compile_patterns()

    def _build_accent_maps(self) -> None:
        accent_patterns = (
            'a*i a*o e*i ia* ia*o ie* io* iu* '
            'A*I A*O E*I IA* IA*O IE* IO* IU* '
            'o*u ua* ua*i ue* ui* uo* üe* '
            'O*U UA* UA*I UE* UI* UO* ÜE* '
            'A* E* I* O* U* Ü* '
            'a* e* i* o* u* ü*'
        )
        self.accent_map: Dict[str, str] = {}
        stars_array = accent_patterns.split()
        nostars_array = accent_patterns.replace('*', '').split()

        for ref, pattern in zip(nostars_array, stars_array):
            self.accent_map[ref] = pattern

    def _compile_patterns(self) -> None:
        self.pinyin_regex = re.compile(f"{self.PINYIN_PATTERN}r?[1-5]", re.IGNORECASE)

    def convert(self, string: str) -> str:
        if not isinstance(string, str) or not string.strip():
            return string

        matches = self.pinyin_regex.findall(string)
        if not matches:
            return string

        for match in matches:
            replacement = self._process_match(match)
            string = string.replace(match, replacement)

        return string

    def _process_match(self, match: str) -> str:
        if len(match) < 2 or not match[-1].isdigit():
            return match

        tone = int(match[-1])
        word = match[:-1].replace('v', 'ü').replace('V', 'Ü')

        for base, pattern in self.accent_map.items():
            if base in word:
                vowel_char = pattern[0]
                accented_vowel = self.TONE_MAP[vowel_char][tone - 1]
                return word.replace(base, pattern).replace(vowel_char, accented_vowel)

        return match

    def reverse_convert(self, string: str) -> str:
        result = []
        syllables = re.findall(self.PINYIN_PATTERN, string, re.IGNORECASE)

        for syl in syllables:
            processed = self._process_syllable(syl)
            result.append(processed)

        return ''.join(result)

    def _process_syllable(self, syllable: str) -> str:
        for i, char in enumerate(syllable):
            if char in self.REVERSE_TONE_MAP:
                base_vowel, tone = self.REVERSE_TONE_MAP[char]
                return f"{syllable[:i]}{base_vowel}{syllable[i + 1:]}{tone}"

        return syllable