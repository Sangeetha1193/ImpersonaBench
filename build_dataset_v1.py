
from __future__ import annotations
import random
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent
DATA = ROOT / "data"
RAW = DATA / "impersonabench_dataset.csv"
GOLD = DATA / "gold100_pairs.csv"
OUT = DATA / "dataset_v1.csv"

COLUMNS = [
    "id", "pair_id", "message", "label", "language", "split_role",
    "source", "is_synthetic", "channel", "role", "persona", "pattern",
    "qc_status", "notes",
]
LABELS = {
    "IMPERSONATION_SCAM",
    "LEGIT_PERSONAL",
    "OTHER_SCAM",
    "LEGIT_TRANSACTIONAL",
}
HINGLISH = re.compile(
    r"\b(hai|hain|hun|bhej|bhejo|kal|abhi|yaar|bhai|didi|papa|mummy|mat|"
    r"nahi|pe|ghar|paisa|jaldi|dunga|karo|mein)\b",
    re.I,
)
VICTIM = re.compile(
    r"^(ok|yes|no|what|maybe|is this|i didn|wait|huh|really|why would|"
    r"should i get)\b",
    re.I,
)
SMS_JUNK = re.compile(
    r"(jaldi kare|abhi kare|please act now)\b", re.I
)
KIN = re.compile(
    r"\b(bhai|didi|papa|mummy|mumma|beta|chacha|cousin|yaar|dost|friend|"
    r"brother|sister|son|daughter|colleague|boss)\b",
    re.I,
)
AUTHORITY = re.compile(
    r"\b(sbi|rbi|otp|kyc|police|aadhaar|irctc|bank|income tax|cibil)\b",
    re.I,
)
CFO_HOLD = re.compile(r"\b(cfo|board meeting|salary advance|rtgs of)\b", re.I)


def lang_of(text: str, fallback: str) -> str | None:
    if re.search(r"[\u0B80-\u0BFF]", text):
        return None
    if re.search(r"[\u0900-\u097F]", text) and not re.search(r"[A-Za-z]{3,}", text):
        return None
    if HINGLISH.search(text):
        return "Hinglish"
    if re.search(r"[A-Za-z]{3,}", text):
        return "English"
    return fallback if fallback in {"English", "Hinglish"} else None


def normalize(text: str) -> str:
    t = str(text).lower().strip()
    t = re.sub(r"₹\s*\d[\d,]*", "AMT", t)
    t = re.sub(r"\b\d[\d,]*\b", "AMT", t)
    t = re.sub(r"[a-z0-9._-]+@[a-z0-9.-]+", "UPI", t)
    t = re.sub(r"[^a-z0-9\u0900-\u097f ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def row(**kwargs) -> dict:
    base = {
        "pair_id": "",
        "persona": "none",
        "pattern": "none",
        "notes": "",
        "channel": "whatsapp",
        "qc_status": "accepted",
        "is_synthetic": True,
        "source": "synthetic_v1",
        "role": "scammer",
        "language": "Hinglish",
        "split_role": "train",
    }
    base.update(kwargs)
    return {k: base[k] for k in COLUMNS if k != "id"}


# --- extra A/B pairs (not in gold100) ---
EXTRA_PAIRS = [
    ("p_101", "phone_broken", "brother", "train",
     "Bhai charger blast ho gaya, yeh naya feature phone hai. 2800 locksmith UPI 9900112233@ybl pe abhi. Call mat karna yeh number unknown.",
     "Bhai locksmith ki zarurat nahi, keys mil gayi. 2800 mat bhejna. Kal tere usual UPI pe coffee ka 200 settle kar dunga."),
    ("p_102", "medical", "parent", "train",
     "Papa MRI slot hold 9500. Technician ke personal QR mri-hold@paytm. Ward se nahi nikal sakta, visiting mat aana.",
     "Papa MRI 9500 hospital radiology counter se prepaid. Technician QR ignore. Report portal pe aa jayegi."),
    ("p_103", "fees", "child", "train",
     "Mumma workshop fee 2100 aaj 4 baje. Coordinator ke UPI workshop@okicici. Class hold, ghar mat aana pehle paisa.",
     "Mumma workshop 2100 school ERP se, last date kal. Coordinator personal UPI nahi chalega."),
    ("p_104", "stranded_friend", "friend", "train",
     "Yaar tyre burst, towing 4100 driver ke UPI tow@ybl. Highway dark, aana mat, sirf UPI. Battery 3%.",
     "Yaar tyre slow puncture, 4100 towing company card se. Driver personal UPI skip. Location share kar raha hun mere number se."),
    ("p_105", "secrecy", "sister", "train",
     "Didi mummy ko mat batana, 4800 chahiye naye UPI 4800@oksbi pe. Private baat, chat delete kar dena.",
     "Didi mummy ke gift 480 ka, mere old UPI pe later. Koi private 4800 nahi, chat rakh sakte ho."),
    ("p_106", "borrowed_phone", "colleague", "train",
     "Using pantry tablet, phone in locker. 1600 snacks UPI pantry-tab@paytm. Don't ping on Teams.",
     "Pantry snacks 1600 expense bot pe. Tablet UPI nahi. Teams pe thread daal diya, phone locker se nikal lunga."),
    ("p_107", "return_tomorrow", "brother", "train",
     "Bhai 3300 abhi 3300@ybl, kal reverse. Naya handle, purana freeze. Didi ko mat bolna.",
     "Bhai 330 jo tune cover kiya, kal tere old UPI pe reverse. Freeze kuch nahi, didi ko bata sakte ho."),
    ("p_108", "phone_broken", "friend", "train",
     "Yaar screen spider crack, cafe wifi se. 1900 repair token 1900@okaxis. Shop uncle gussa, call mat.",
     "Yaar screen crack, repair 1900 kal shop POS pe. Cafe wale ko UPI mat karna. Mera number same hai."),
    ("p_109", "medical", "sister", "eval_holdout",
     "Didi dengue test 3600, lab rider ke UPI rider-lab@ybl. Fasting toot jayega warna, unko ring mat karna.",
     "Didi dengue test 3600 lab window pe jaise last CBC. Rider UPI skip. Breakfast ke baad jaungi."),
    ("p_110", "fees", "parent", "eval_holdout",
     "Papa smart class fee 2700 teacher GPay 2700@paytm aaj only. PTM start, disturb mat karna.",
     "Papa smart class 2700 circular wale school account se. Teacher GPay pe fee mat bhejna."),
    ("p_111", "stranded_friend", "brother", "train",
     "Bhai metro last train miss, lodge 2400 UPI lodge-temp@oksbi. Borrowed phone, call drop hoga.",
     "Bhai last metro miss, lodge 2400 desk card se. Temp UPI nahi. Subah wali first train le lunga."),
    ("p_112", "borrowed_phone", "parent", "train",
     "Mummy chemist ke counter phone se. 850 medicine UPI chemist-temp@ybl, queue lambi, aana mat.",
     "Mummy medicine 850 pharmacy bill pe card. Chemist personal UPI nahi. Queue se nikal ke call karungi."),
    ("p_113", "return_tomorrow", "colleague", "train",
     "Need 1400 on 1400@paytm now for cab, refund tomorrow. Wallet KYC failed. Skip standup notes.",
     "Cab 1400 Uber business. Personal Paytm refund ki need nahi. Standup mein optional mention."),
    ("p_114", "secrecy", "child", "train",
     "Papa yeh screenshot mat lena. 5600 temp UPI pe sports kit private. Coach ko call mat karna.",
     "Papa sports kit 5600 school listed vendor se. Private temp UPI nahi. Screenshot receipt ke liye rakhna theek hai."),
    ("p_115", "phone_broken", "sister", "train",
     "Didi handset pawn shop mein, yeh naya number. 3100 release UPI pawn@okicici. Location mat poochna.",
     "Didi phone theek hai, pawn kuch nahi. 3100 mat bhejna. Kal usual UPI pe shopping split 300 kar dena."),
    ("p_116", "medical", "friend", "train",
     "Yaar stitches extra 2200, intern ke QR intern-med@paytm. Blood dekh ke aana mat, sirf paisa.",
     "Yaar stitches ho gaye, 2200 official discharge bill mein. Intern QR ignore. Photo invoice bhej raha hun."),
    ("p_117", "fees", "brother", "train",
     "Bhai convocation fee 1800 portal down, student UPI 1800@ybl midnight tak. Admin mail mat kar.",
     "Bhai convocation 1800 kal official window. Student UPI skip, portal screenshot rakh."),
    ("p_118", "stranded_friend", "friend", "eval_holdout",
     "Yaar bus stand agent last ticket 1700 UPI 1700@ybl. Stand ke peeche, aana mat.",
     "Yaar waitlist official window se. Agent UPI skip. Next bus bhi option hai, 1700 panic nahi."),
    ("p_119", "borrowed_phone", "friend", "train",
     "Guard tablet se, 900 parcel COD UPI guard-cod@paytm. Gate wait, bahar mat aana.",
     "Parcel Amazon pe already paid. Guard COD UPI skip. Gate pe 2 min, mera number se."),
    ("p_120", "return_tomorrow", "friend", "eval_holdout",
     "Yaar 2600 now 2600@oksbi I'll return tomorrow after cash out. Don't save beneficiary, one-time UPI.",
     "Yaar 2600 jo tune cover kiya kal tere saved UPI pe wapas. One-time handle mat banana, old ID use karunga."),
]

OTHER_SCAM = [
    ("Aapka KYC pending hai. Verify karein is link pe sbi-kyc-now.co.in warna account block.", "Hinglish", "kyc", "sms"),
    ("Dear customer, KYC incomplete. Update at hdfc-verify-kyc.net to avoid debit freeze.", "English", "kyc", "sms"),
    ("Income Tax refund Rs 12450 approved. Share OTP 847392 to credit the amount.", "English", "otp", "sms"),
    ("SBI: suspicious txn Rs 49999. Share the OTP you received to reverse it in 2 minutes.", "English", "otp", "sms"),
    ("Aapka ATM card block ho gaya hai. Details update karein is number pe OTP bhejo.", "Hinglish", "otp", "sms"),
    ("RBI escrow: transfer all funds for digital arrest verification, refund in 30 minutes.", "English", "digital_arrest", "whatsapp"),
    ("Police cyber cell: Aadhaar linked to crime. Call now, share OTP or warrant in 24 hours.", "English", "digital_arrest", "sms"),
    ("You won a lucky draw of Rs 25 lakh. Pay Rs 1500 GST on UPI lucky-win@ybl to claim.", "English", "lottery", "sms"),
    ("Aap lottery jeet gaye. Claim karne ke liye 999 processing UPI pe bhejo prize@paytm.", "Hinglish", "lottery", "whatsapp"),
    ("Flipkart refund Rs 1200 stuck. Accept this UPI collect request to receive it today.", "English", "refund", "sms"),
    ("Sir refund lene ke liye UPI request accept kar do, warna amount lapse ho jayega.", "Hinglish", "refund", "whatsapp"),
    ("Your loan is pre-approved Rs 3 lakh. Pay Rs 699 GST to 699@oksbi to disburse.", "English", "loan", "sms"),
    ("Bhai mein Ankit from CashNow. Loan approved, pay 699 GST to my UPI then 30k aayega.", "Hinglish", "loan", "whatsapp"),
    ("IRCTC refund pending. Claim at https://irctc-railmadad.gov-india.ml before departure.", "English", "phishing", "sms"),
    ("Meta Verified India Rs 299/month. Pay UPI meta-in@okaxis to get blue tick today.", "English", "phishing", "whatsapp"),
    ("Your electricity will disconnect in 1 hour. Pay on bescom-pay.tk not the official app.", "English", "utilities", "sms"),
    ("Bijli bill pending, is short link pe pay karo warna connection cut. Official app mat use.", "Hinglish", "utilities", "sms"),
    ("Crypto platform making Rs 8 lakh/month. Start with Rs 30000 on this UPI to join.", "English", "investment", "whatsapp"),
    ("Dost ne MLM bataya, 20000 dalo 3 mahine mein double. UPI id scheme@ybl pe bhejo.", "Hinglish", "investment", "whatsapp"),
    ("Amazon: your account will be closed. Click amzn-secure-login.co to verify card.", "English", "phishing", "sms"),
    ("Parcel customs me atka hai, 35 pay karke release kare on customs-pay.tk.", "Hinglish", "phishing", "sms"),
    ("UPI Circle: friend added you as secondary. Accept at upi-circle-accept.npci-in.click.", "English", "phishing", "whatsapp"),
    ("Bank manager Sharma: dubious deposit Rs 1.2 lakh. Reverse ke liye OTP share karein.", "Hinglish", "kyc", "sms"),
    ("Dear customer your debit card will be blocked today due to KYC expiry. Share OTP.", "English", "kyc", "sms"),
    ("Navi loan photo + Aadhaar WhatsApp pe bhejo 1 ghante mein paisa. OTP bhi forward karo.", "Hinglish", "loan", "whatsapp"),
    ("GSTN: file now on gst-india-filing.tk using OTP sent to you, ignore gst.gov.in.", "English", "phishing", "sms"),
    ("Paytm KYC failed. Re-kyc pe 1 rupee test UPI accept karo warna wallet freeze.", "Hinglish", "kyc", "sms"),
    ("Your PAN is linked to a mule account. Pay Rs 8500 fine via UPI pan-fine@okaxis.", "English", "digital_arrest", "sms"),
    ("Aapka ration card cancel. Update pe OTP is number pe bhejo, link ration-kyc.tk.", "Hinglish", "kyc", "sms"),
    ("Credit card limit increased. Confirm CVV + OTP on hdfc-limit-up.co.in immediately.", "English", "otp", "sms"),
    ("BGMI Mythic skin Rs 500, account transfer after UPI to bgmi-skin-trade@okbizaxis.", "English", "investment", "whatsapp"),
    ("WhatsApp: your account ban in 24h. Verify on wa-support-in.tk with SMS code.", "English", "phishing", "whatsapp"),
    ("EPFO claim pending. Submit UAN OTP on epfo-claim-in.net to release Rs 42000.", "English", "otp", "sms"),
    ("Aapke naam FIR. Court fee 15000 RBI escrow UPI escrow-rbi@ybl pe warna arrest.", "Hinglish", "digital_arrest", "whatsapp"),
    ("Netflix India sharing crackdown. Pay 199 on nflix-pay@ybl to keep your family slot.", "English", "phishing", "sms"),
    ("Aapka Fastag blacklisted. Recharge 500 on fastag-help.tk not NHAI app.", "Hinglish", "phishing", "sms"),
    ("Job offer WFH Rs 45000. Pay Rs 999 registration UPI hr-onboard@oksbi.", "English", "loan", "whatsapp"),
    ("Aap select hue part time. 999 security deposit UPI pe, kaam kal se.", "Hinglish", "loan", "whatsapp"),
    ("Axis alert: unauthorized login Delhi. Share OTP to block, else Rs 2 lakh debit.", "English", "otp", "sms"),
    ("Aapka UPI PIN expire. Reset pe yeh OTP forward karo support number pe.", "Hinglish", "otp", "sms"),
]

LEGIT_TX = [
    ("Your payment of Rs 899 to ABC Store was successful. If not you, call the number on the back of your card.", "English", "sms"),
    ("Aapka Rs 899 ka payment ABC Store pe successful raha. Card back pe number pe call karo if not you.", "Hinglish", "sms"),
    ("Your order has been shipped and will arrive tomorrow. Track in the Amazon app only.", "English", "sms"),
    ("Order ship ho gaya, kal deliver. Tracking Amazon app se, kisi SMS link pe click mat karna.", "Hinglish", "sms"),
    ("SBI: Rs 2000 debited to merchant Flipkart. If not you, block card in YONO.", "English", "sms"),
    ("SBI: 2000 Flipkart pe debit. Agar aapne nahi kiya toh YONO se card block karo.", "Hinglish", "sms"),
    ("Your OTP is 456789 for netbanking login. Do not share it with anyone.", "English", "sms"),
    ("Netbanking OTP 456789 hai. Kisi ko mat batana, bank kabhi OTP nahi maangta.", "Hinglish", "sms"),
    ("BESCOM: bill Rs 3840 due 25-Apr. Pay at bescom.org or authorised centres only.", "English", "sms"),
    ("Bijli bill 3840, 25 Apr tak bescom.org pe. SMS wale random link pe mat bharna.", "Hinglish", "sms"),
    ("Indane LPG booking IND28475 confirmed. Share OTP only with the delivery person at the door.", "English", "sms"),
    ("Indane booking confirm. OTP sirf delivery pe door par dena, call pe nahi.", "Hinglish", "sms"),
    ("Your train ticket has been booked successfully. PNR is in the IRCTC app.", "English", "sms"),
    ("Train ticket book ho gayi. PNR IRCTC app mein hai, koi UPI request accept mat karna.", "Hinglish", "sms"),
    ("UIDAI: book Aadhaar update at myaadhaar.uidai.gov.in only. OTP comes after you book.", "English", "sms"),
    ("Aadhaar update sirf myaadhaar.uidai.gov.in. Pehle OTP kisi ko mat bhejna.", "Hinglish", "sms"),
    ("CIBIL score updated to 782. Check at cibil.com with your registered mobile.", "English", "sms"),
    ("CIBIL 782 update. cibil.com pe registered mobile se dekho, UPI collect ignore.", "Hinglish", "sms"),
    ("GSTR-3B due 20-Apr. File at gst.gov.in only. We will never ask for OTP on SMS.", "English", "sms"),
    ("GSTR due 20 Apr, gst.gov.in. SMS pe OTP maange toh mat dena.", "Hinglish", "sms"),
    ("HDFC: no pending RTGS on your account. Official helpline is 1860-266-0333 only.", "English", "sms"),
    ("HDFC: koi pending RTGS nahi. Helpline 1860-266-0333, random number pe mat call.", "Hinglish", "sms"),
    ("Thank you for your purchase at ABC Store. Receipt is in your email.", "English", "sms"),
    ("Purchase ke liye thanks, receipt email pe hai. Extra UPI request ignore karo.", "Hinglish", "sms"),
    ("Your refund of Rs 1200 is being processed. Credit in 3-5 days to the original account.", "English", "sms"),
    ("Refund 1200 original account mein 3-5 din. Collect request accept mat karna.", "Hinglish", "sms"),
    ("Flipkart: delivery today 4-6pm. OTP will be asked by the delivery partner at the door.", "English", "sms"),
    ("Flipkart aaj 4-6. OTP sirf door pe partner ko, call pe nahi.", "Hinglish", "sms"),
    ("Airtel: recharge Rs 299 successful. Validity 28 days. No extra action needed.", "English", "sms"),
    ("Airtel recharge 299 successful, 28 din. Koi extra UPI nahi bhejna.", "Hinglish", "sms"),
    ("ICICI: Rs 500 credited from salary. Available balance in iMobile, no UPI request to confirm.", "English", "sms"),
    ("Salary 500 credit hui iMobile pe. Confirm ke liye koi UPI collect mat accept karna.", "Hinglish", "sms"),
    ("Your cab ride with Uber is complete. Rs 247 was charged to your saved card ending 4412.", "English", "sms"),
    ("Uber ride 247 saved card se kata. Driver ko extra UPI tabhi do jab app me tip ho.", "Hinglish", "sms"),
    ("JioFiber bill Rs 699 auto-pay scheduled. Manage in the MyJio app.", "English", "sms"),
    ("JioFiber 699 auto-pay MyJio se. SMS link se payment mat karna.", "Hinglish", "sms"),
    ("Swiggy order delivered. Rate the order in the app. We will not call for UPI.", "English", "sms"),
    ("Swiggy deliver ho gaya. App me rating. Call pe UPI maange toh mat bhejna.", "Hinglish", "sms"),
    ("EPFO: passbook updated. View only on umang.gov.in or the UMANG app.", "English", "sms"),
    ("EPFO passbook update. Sirf UMANG / umang.gov.in. Bahar OTP mat do.", "Hinglish", "sms"),
    ("IRCTC: ticket booked PNR 4218761234. Cancel only in the IRCTC app.", "English", "sms"),
    ("IRCTC ticket book, PNR app me. Refund ke liye random link mat kholna.", "Hinglish", "sms"),
    ("PhonePe: you paid Rs 150 to a trusted payee. If this wasn't you, open the app > help.", "English", "sms"),
    ("PhonePe 150 trusted payee ko. Agar aap nahi the toh app > help, SMS pe reply mat karo.", "Hinglish", "sms"),
]


def extra_pair_rows() -> list[dict]:
    rows = []
    for pid, pattern, persona, split, scam, legit in EXTRA_PAIRS:
        shared = dict(
            pair_id=pid, pattern=pattern, persona=persona, split_role=split,
            language="Hinglish", source="synthetic_v1", is_synthetic=True,
            channel="whatsapp", qc_status="accepted",
            notes="v1 extra matched pair",
        )
        rows.append(row(**shared, message=scam, label="IMPERSONATION_SCAM", role="scammer"))
        rows.append(row(**shared, message=legit, label="LEGIT_PERSONAL", role="personal"))
    return rows


def list_rows(items, label, role, pattern_key) -> list[dict]:
    out = []
    n = len(items)
    hold_n = max(1, round(n * 0.2))
    for i, item in enumerate(items):
        msg, language = item[0], item[1]
        pattern = item[2] if len(item) > 2 and label == "OTHER_SCAM" else pattern_key
        channel = item[-1] if item[-1] in {"sms", "whatsapp"} else "sms"
        split = "eval_holdout" if i >= n - hold_n else "train"
        out.append(row(
            pair_id="", message=msg, label=label, language=language,
            split_role=split, source="synthetic_v1", is_synthetic=True,
            channel=channel, role=role, persona="none", pattern=pattern,
            qc_status="accepted", notes="synthetic control class",
        ))
    return out


def remap_public() -> list[dict]:
    raw = pd.read_csv(RAW)
    kept = []
    for _, r in raw.iterrows():
        msg = str(r["message"]).strip()
        reason = str(r.get("reason", "")).strip()
        old_label = str(r.get("label", "")).lower()
        old_lang = str(r.get("language", "English"))
        if not msg or VICTIM.match(msg) and len(msg) < 90:
            continue
        if CFO_HOLD.search(msg):
            continue
        if SMS_JUNK.search(msg):
            continue
        language = lang_of(msg, old_lang)
        if language is None:
            continue

        rl = reason.lower()
        label = None
        pattern = "public_remap"
        persona = "none"
        role = "scammer"
        channel = "sms"

        if rl in {"trusted-contact impersonation"} or (
            rl == "impersonation" and KIN.search(msg) and not AUTHORITY.search(msg)
        ):
            label, pattern, role, channel = "IMPERSONATION_SCAM", "trusted_contact", "scammer", "whatsapp"
            persona = "friend"
        elif rl in {
            "kyc_fraud", "otp_theft", "loan_app_fraud", "investment_fraud",
            "fake refund request", "fake bank verification", "fake account verification",
            "urgency, reward or suspicious instruction",
        }:
            label, pattern, role, channel = "OTHER_SCAM", rl.replace(" ", "_")[:40], "scammer", "sms"
        elif rl == "impersonation":
            label, pattern = "OTHER_SCAM", "brand_or_authority"
        elif rl in {"normal family conversation", "normal friend conversation",
                    "normal personal conversation", "normal repayment conversation"}:
            label, pattern, role, channel = "LEGIT_PERSONAL", "legit_personal", "personal", "whatsapp"
        elif rl in {"normal transactional message", "normal transaction notification", "benign"}:
            label, pattern, role, channel = "LEGIT_TRANSACTIONAL", "transactional", "institution", "sms"
        elif rl == "borderline":
            if re.search(r"share otp|upi collect|click .*http", msg, re.I):
                label, pattern = "OTHER_SCAM", "borderline_re_scam"
            else:
                continue
        elif rl == "urgent money request":
            label = "IMPERSONATION_SCAM" if KIN.search(msg) else "OTHER_SCAM"
            pattern = "urgent_money"
        elif old_label == "scam":
            label = "OTHER_SCAM"
        elif old_label == "benign":
            label = "LEGIT_TRANSACTIONAL"
            role, channel = "institution", "sms"
        else:
            continue

        kept.append(row(
            pair_id="", message=msg, label=label, language=language,
            split_role="train", source="public_remap", is_synthetic=False,
            channel=channel, role=role, persona=persona, pattern=pattern,
            qc_status="accepted", notes=f"remap:{reason}",
        ))
    return kept


def load_gold() -> list[dict]:
    df = pd.read_csv(GOLD)
    rows = []
    for _, r in df.iterrows():
        rows.append(row(
            pair_id=r["pair_id"], message=r["message"], label=r["label"],
            language="Hinglish", split_role=r["split_role"],
            source="synthetic_v1", is_synthetic=True, channel="whatsapp",
            role=r["role"], persona=r["persona"], pattern=r["pattern"],
            qc_status="accepted", notes="gold100 seed",
        ))
    return rows


def dedup(rows: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for r in rows:
        key = normalize(r["message"])
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def assign_holdout_unpaired(rows: list[dict]) -> None:
    """Keep pair splits; unpaired rows get ~20% holdout per label."""
    by_label: dict[str, list[dict]] = {}
    for r in rows:
        if r["pair_id"]:
            continue
        by_label.setdefault(r["label"], []).append(r)
    for items in by_label.values():
        n_hold = max(1, round(len(items) * 0.2))
        hold = [x for x in items if x["split_role"] == "eval_holdout"]
        train = [x for x in items if x["split_role"] != "eval_holdout"]
        while len(hold) < n_hold and train:
            x = train.pop()
            x["split_role"] = "eval_holdout"
            hold.append(x)


def qc(df: pd.DataFrame) -> dict:
    issues = []
    if set(df["label"]) - LABELS:
        issues.append(f"bad labels {set(df['label']) - LABELS}")
    if df["message"].map(normalize).duplicated().any():
        issues.append("normalized duplicates remain")
    train = df[df["split_role"] == "train"]["message"].map(normalize)
    hold = df[df["split_role"] == "eval_holdout"]["message"].map(normalize)
    leak = set(train) & set(hold)
    if leak:
        issues.append(f"exact split leakage n={len(leak)}")
    for pid, g in df[df["pair_id"].astype(str).str.len() > 0].groupby("pair_id"):
        if g["split_role"].nunique() != 1:
            issues.append(f"pair split mismatch {pid}")
    return {
        "n": len(df),
        "labels": df["label"].value_counts().to_dict(),
        "languages": df["language"].value_counts().to_dict(),
        "splits": df["split_role"].value_counts().to_dict(),
        "sources": df["source"].value_counts().to_dict(),
        "issues": issues,
    }


def main() -> None:
    rows = []
    rows.extend(load_gold())
    rows.extend(extra_pair_rows())
    rows.extend(list_rows(OTHER_SCAM, "OTHER_SCAM", "scammer", "other_scam"))
    rows.extend(list_rows(LEGIT_TX, "LEGIT_TRANSACTIONAL", "institution", "transactional"))
    remapped = remap_public()
    extra_scam = [r for r in remapped if r["label"] == "OTHER_SCAM"]
    random.Random(42).shuffle(extra_scam)
    extra_scam = extra_scam[:90]
    other_remap = [r for r in remapped if r["label"] != "OTHER_SCAM"]
    rows.extend(other_remap)
    rows.extend(extra_scam)
    rows = dedup(rows)
    assign_holdout_unpaired(rows)

    df = pd.DataFrame(rows)
    df.insert(0, "id", [f"ib_v1_{i:06d}" for i in range(1, len(df) + 1)])
    df = df[COLUMNS]
    report = qc(df)
    df.to_csv(OUT, index=False)
    print("Wrote", OUT)
    print("rows", report["n"])
    print("labels", report["labels"])
    print("languages", report["languages"])
    print("splits", report["splits"])
    print("sources", report["sources"])
    print("qc_issues", report["issues"] or "none")
    if report["issues"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
