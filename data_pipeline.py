import time
import random
import re
from abc import ABC, abstractmethod
from functools import wraps

import pandas as pd
from Datasets import load_dataset


def log_step(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print("-" * 50)
        print(f"Starting: {func.__name__}")
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Finished: {func.__name__}")
        print(f"Execution Time: {end - start:.2f} seconds")
        print("-" * 50)
        return result

    return wrapper


class BasePipeline(ABC):
    @abstractmethod
    def run(self):
        pass


class Downloader(BasePipeline):
    def __init__(self, data_sets):
        self.data_sets = data_sets

    @log_step
    def download_dataset(self, dataset_name):
        print(dataset_name)
        ds = load_dataset(dataset_name)
        first_split = list(ds.keys())[0]
        return ds[first_split].to_pandas()

    @log_step
    def run(self):
        print("Downloading...")
        downloaded_data = {}
        for dataset in self.data_sets:
            try:
                downloaded_data[dataset] = self.download_dataset(dataset)
            except Exception as e:
                print(f"Failed to download {dataset}")
                print(e)
        return downloaded_data


class Standardizer(BasePipeline):
    REQUIRED_COLUMNS = ["message", "label", "reason", "domain", "language"]

    LABEL_MAP = {
        "scam": "scam",
        "fraud": "scam",
        "spam": "scam",
        "legit": "benign",
        "legitimate": "benign",
        "benign": "benign",
        "ham": "benign",
        "normal": "benign",
    }

    LANGUAGE_MAP = {
        "en": "English",
        "eng": "English",
        "english": "English",
        "hi": "Hindi",
        "hin": "Hindi",
        "hindi": "Hindi",
        "hinglish": "Hinglish",
        "ta": "Tamil",
        "tam": "Tamil",
        "tamil": "Tamil",
        "bn": "Bengali",
        "bengali": "Bengali",
        "kn": "Kannada",
        "kannada": "Kannada",
        "te": "Telugu",
        "telugu": "Telugu",
        "mr": "Marathi",
        "marathi": "Marathi",
    }

    def __init__(self, downloaded_data):
        self.downloaded_data = downloaded_data

    def _find_text_column(self, df):
        possible_columns = ["message", "text", "sms", "content"]
        lower_map = {str(col).lower().strip(): col for col in df.columns}
        for name in possible_columns:
            if name in lower_map:
                return lower_map[name]
        raise ValueError(f"No text column found in columns: {list(df.columns)}")

    def _normalize_label(self, value):
        if pd.isna(value):
            return None
        if isinstance(value, bool):
            return "scam" if value else "benign"
        key = str(value).strip().lower()
        return self.LABEL_MAP.get(key, key)

    def _normalize_language(self, value):
        if pd.isna(value):
            return "unknown"
        key = str(value).strip().lower()
        return self.LANGUAGE_MAP.get(key, str(value).strip().title())

    def _as_dict(self, value):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return {}
        if isinstance(value, dict):
            return value
        try:
            return dict(value)
        except Exception:
            return {}

    def _as_list(self, value):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return []
        if isinstance(value, list):
            return value
        try:
            return list(value)
        except Exception:
            return []

    def _flatten_chakravyuh(self, df):
        rows = []
        for _, row in df.iterrows():
            ground_truth = self._as_dict(row.get("ground_truth"))
            metadata = self._as_dict(row.get("metadata"))
            source = self._as_dict(row.get("source"))
            attack_sequence = self._as_list(row.get("attack_sequence"))

            is_scam = bool(ground_truth.get("is_scam", True))
            label = "scam" if is_scam else "benign"
            reason = ground_truth.get("category") or "Unknown"
            domain = source.get("category") or "finance"
            language = self._normalize_language(metadata.get("language", "unknown"))

            for turn in attack_sequence:
                turn = self._as_dict(turn)
                text = turn.get("text", "")
                if not text:
                    continue
                turn_lang = turn.get("language")
                rows.append(
                    {
                        "message": text,
                        "label": label,
                        "reason": reason,
                        "domain": domain,
                        "language": self._normalize_language(turn_lang)
                        if turn_lang
                        else language,
                    }
                )

        return pd.DataFrame(rows)

    def _standardize_one(self, df, dataset_name=""):
        df = df.copy()

        if "attack_sequence" in df.columns:
            df = self._flatten_chakravyuh(df)
        else:
            text_column = self._find_text_column(df)
            df = df.rename(columns={text_column: "message"})

            lower_map = {str(col).lower().strip(): col for col in df.columns}
            for target in ("label", "reason", "domain", "language"):
                if target not in df.columns and target in lower_map:
                    df = df.rename(columns={lower_map[target]: target})

            if "label" not in df.columns:
                df["label"] = None
            if "reason" not in df.columns:
                df["reason"] = "Unknown"
            if "domain" not in df.columns:
                df["domain"] = "general"
            if "language" not in df.columns:
                df["language"] = "unknown"

        df["label"] = df["label"].apply(self._normalize_label)
        df["language"] = df["language"].apply(self._normalize_language)
        df["reason"] = df["reason"].fillna("Unknown").astype(str)
        df["domain"] = df["domain"].fillna("general").astype(str)

        return df[self.REQUIRED_COLUMNS]

    @log_step
    def run(self):
        standardized_data = {}
        for dataset_name, df in self.downloaded_data.items():
            print(f"Standardizing: {dataset_name}")
            standardized_data[dataset_name] = self._standardize_one(df, dataset_name)
        return standardized_data


class Cleaner(BasePipeline):
    def __init__(self, standardized_data):
        self.standardized_data = standardized_data

    def _clean_text(self, text):
        if pd.isna(text):
            return ""
        text = str(text).strip()
        return re.sub(r"\s+", " ", text)

    def _clean_one(self, df):
        df = df.copy()
        df["message"] = df["message"].apply(self._clean_text)
        df = df[df["message"] != ""]
        df = df.drop_duplicates(subset=["message"])
        return df.reset_index(drop=True)

    @log_step
    def run(self):
        cleaned_data = {}
        for dataset_name, df in self.standardized_data.items():
            print(f"Cleaning: {dataset_name}")
            cleaned_data[dataset_name] = self._clean_one(df)
        return cleaned_data


class Validator(BasePipeline):
    REQUIRED_COLUMNS = ["message", "label", "reason", "domain", "language"]
    VALID_LABELS = {"scam", "benign"}
    VALID_LANGUAGES = {
        "English",
        "Hinglish",
        "Hindi",
        "Tamil",
        "Bengali",
        "Kannada",
        "Telugu",
        "Marathi",
        "unknown",
    }

    def __init__(self, cleaned_data):
        self.cleaned_data = cleaned_data

    def _validate_columns(self, df):
        missing = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

    def _validate_labels(self, df):
        invalid_labels = set(df["label"].dropna()) - self.VALID_LABELS
        if invalid_labels:
            raise ValueError(f"Invalid labels found: {invalid_labels}")

    def _validate_languages(self, df):
        invalid_languages = set(df["language"].dropna()) - self.VALID_LANGUAGES
        if invalid_languages:
            raise ValueError(f"Invalid languages found: {invalid_languages}")

    def _validate_empty_messages(self, df):
        if df["message"].isna().any() or (df["message"].astype(str).str.strip() == "").any():
            raise ValueError("Dataset contains empty messages")

    def _validate_one(self, df):
        self._validate_columns(df)
        self._validate_labels(df)
        self._validate_languages(df)
        self._validate_empty_messages(df)
        return True

    @log_step
    def run(self):
        for dataset_name, df in self.cleaned_data.items():
            print(f"Validating: {dataset_name}")
            self._validate_one(df)
            print(f"OK {dataset_name} is valid")
        return self.cleaned_data


class Synthesizer(BasePipeline):
    def __init__(self, samples_per_language=100):
        self.samples_per_language = samples_per_language
        self.templates = {
            "English": {
                "scam": [
                    (
                        "Your {service} account will be blocked. "
                        "Please verify immediately using the link sent to you.",
                        "Fake account verification",
                    ),
                    (
                        "Your refund of ₹{amount} is pending. "
                        "Approve the payment request to receive it.",
                        "Fake refund request",
                    ),
                    (
                        "I am your {persona}. I need ₹{amount} urgently. "
                        "Please transfer it now.",
                        "Trusted-contact impersonation",
                    ),
                ],
                "benign": [
                    (
                        "Your payment of ₹{amount} to {merchant} was successful.",
                        "Normal transaction notification",
                    ),
                    (
                        "I will send you ₹{amount} tomorrow.",
                        "Normal repayment conversation",
                    ),
                    (
                        "Let's meet for dinner tonight.",
                        "Normal personal conversation",
                    ),
                ],
            },
            "Hinglish": {
                "scam": [
                    (
                        "Bhai, mujhe ₹{amount} urgently chahiye. "
                        "Abhi UPI kar de, kal return kar dunga.",
                        "Trusted-contact impersonation",
                    ),
                    (
                        "Aapka bank account verification pending hai. "
                        "Abhi verify karo warna account block ho jayega.",
                        "Fake bank verification",
                    ),
                    (
                        "Sir, refund lene ke liye UPI request accept kar do.",
                        "Fake refund request",
                    ),
                ],
                "benign": [
                    (
                        "Bhai shaam ko chai pe milte hain.",
                        "Normal friend conversation",
                    ),
                    (
                        "Main kal ₹{amount} UPI se bhej dunga.",
                        "Normal repayment conversation",
                    ),
                    (
                        "Aaj ghar pe dinner karenge.",
                        "Normal family conversation",
                    ),
                ],
            },
            "Tamil": {
                "scam": [
                    (
                        "உங்கள் வங்கி கணக்கு விரைவில் முடக்கப்படும். "
                        "உடனே verification செய்யுங்கள்.",
                        "Fake bank verification",
                    ),
                    (
                        "₹{amount} refund pending-ஆ இருக்கு. "
                        "இந்த UPI request-ஐ approve பண்ணுங்க.",
                        "Fake refund request",
                    ),
                    (
                        "எனக்கு ₹{amount} அவசரமாக தேவை. "
                        "இப்பவே UPI அனுப்புங்க.",
                        "Urgent money request",
                    ),
                ],
                "benign": [
                    (
                        "இன்று இரவு வீட்டில் dinner தயார் பண்ணுங்க.",
                        "Normal family conversation",
                    ),
                    (
                        "நாளைக்கு ₹{amount} UPI மூலம் அனுப்புகிறேன்.",
                        "Normal repayment conversation",
                    ),
                    (
                        "இன்று மாலை சந்திப்போமா?",
                        "Normal personal conversation",
                    ),
                ],
            },
        }
        self.personas = ["brother", "friend", "manager", "colleague"]
        self.services = ["bank", "UPI", "wallet"]
        self.merchants = ["ABC Store", "Amazon", "Flipkart"]

    def _render_template(self, template, reason, language, label):
        amount = random.choice([200, 500, 1000, 1500, 2000, 3000])
        persona = random.choice(self.personas)
        service = random.choice(self.services)
        merchant = random.choice(self.merchants)
        message = template.format(
            amount=amount,
            persona=persona,
            service=service,
            merchant=merchant,
        )
        return {
            "message": message,
            "label": label,
            "reason": reason,
            "domain": "finance" if label == "scam" else "general",
            "language": language,
        }

    def _generate_language(self, language):
        rows = []
        for label in ["scam", "benign"]:
            templates = self.templates[language][label]
            for _ in range(self.samples_per_language // 2):
                template, reason = random.choice(templates)
                rows.append(
                    self._render_template(template, reason, language, label)
                )
        return rows

    @log_step
    def run(self):
        rows = []
        for language in self.templates:
            print(f"Generating {language} samples...")
            rows.extend(self._generate_language(language))
        return pd.DataFrame(rows)


class Deduplicator(BasePipeline):
    def __init__(self, data):
        self.data = data

    def _normalize_text(self, text):
        text = str(text).strip().lower()
        return re.sub(r"\s+", " ", text)

    def _remove_exact_duplicates(self, df):
        df = df.copy()
        df["_normalized_message"] = df["message"].apply(self._normalize_text)
        df = df.drop_duplicates(subset=["_normalized_message"])
        df = df.drop(columns=["_normalized_message"])
        return df.reset_index(drop=True)

    @log_step
    def run(self):
        print(f"Before deduplication: {len(self.data)}")
        deduplicated_data = self._remove_exact_duplicates(self.data)
        print(f"After deduplication: {len(deduplicated_data)}")
        print(f"Removed: {len(self.data) - len(deduplicated_data)}")
        return deduplicated_data


class Merger(BasePipeline):
    REQUIRED_COLUMNS = ["message", "label", "reason", "domain", "language"]

    def __init__(self, real_data, synthetic_data):
        self.real_data = real_data
        self.synthetic_data = synthetic_data

    def _validate_schema(self, df):
        if list(df.columns) != self.REQUIRED_COLUMNS:
            raise ValueError(f"Unexpected columns: {list(df.columns)}")

    @log_step
    def run(self):
        frames = []
        for name, df in self.real_data.items():
            print(f"Merging real dataset: {name}")
            self._validate_schema(df)
            frames.append(df)

        self._validate_schema(self.synthetic_data)
        frames.append(self.synthetic_data)

        merged_data = pd.concat(frames, ignore_index=True)
        print(f"Total rows: {len(merged_data)}")
        return merged_data


def main():
    datasets = [
        "karanverma19/Indian_Multilingual_Scam_Message_Dataset",
        "ujjwalpardeshi/chakravyuh-bench-v0",
    ]

    downloaded_data = Downloader(datasets).run()
    if not downloaded_data:
        raise RuntimeError("No datasets were downloaded successfully.")

    standardized_data = Standardizer(downloaded_data).run()
    cleaned_data = Cleaner(standardized_data).run()

    for name, df in cleaned_data.items():
        print("\nDataset:", name)
        print("Rows:", len(df))
        print("Duplicates:", df["message"].duplicated().sum())
        print(df.head())

    validated_data = Validator(cleaned_data).run()

    synthetic_data = Synthesizer(samples_per_language=100).run()
    print(synthetic_data.head())
    print(synthetic_data["language"].value_counts())
    print(synthetic_data["label"].value_counts())

    deduplicated_synthetic = Deduplicator(synthetic_data).run()
    merged_data = Merger(validated_data, deduplicated_synthetic).run()
    final_data = Deduplicator(merged_data).run()

    output_path = "impersonabench_dataset.csv"
    final_data.to_csv(output_path, index=False)
    print(f"Saved final dataset to {output_path} ({len(final_data)} rows)")


if __name__ == "__main__":
    main()
