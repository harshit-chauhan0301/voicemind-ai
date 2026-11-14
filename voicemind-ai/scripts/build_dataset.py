"""Generate a rich 5-domain sentiment dataset: python -m scripts.build_dataset"""
from __future__ import annotations

import csv
import os
import random
import stat
from pathlib import Path

from utils.config import SENTIMENT_DATASET_PATH

SUBJECTS = [
    "app", "book", "camera", "car", "class", "concert", "course", "delivery",
    "game", "hotel", "laptop", "meal", "movie", "phone", "product",
    "restaurant", "service", "teacher", "update", "website",
    "voice studio", "summarizer", "analyzer", "microphone", "transcription",
    "interface", "dashboard", "system", "model", "feature", "audio quality",
    "customer support", "team", "platform", "software", "tool", "experience",
]

POS_ADJS = [
    "amazing", "awesome", "beautiful", "best", "brilliant", "charming",
    "clean", "clear", "comfortable", "cool", "crisp", "delicious", "delightful",
    "easy to use", "efficient", "enjoyable", "excellent", "exceptional",
    "fabulous", "fantastic", "fast", "flawless", "friendly", "fun", "good",
    "great", "helpful", "impressive", "incredible", "intuitive", "kind",
    "lifesaving", "lovely", "nice", "outstanding", "perfect", "phenomenal",
    "pleasant", "polite", "powerful", "quick", "reliable", "responsive",
    "satisfying", "seamless", "sharp", "smart", "smooth", "solid", "spot on",
    "stellar", "super", "superb", "supportive", "terrific", "thrilling",
    "top notch", "useful", "valuable", "wonderful", "worthwhile",
]

NEG_ADJS = [
    "a disaster", "a joke", "a letdown", "a mess", "a nightmare", "a waste",
    "annoying", "atrocious", "awful", "bad", "boring", "broken", "buggy",
    "cheap and flimsy", "clumsy", "clunky", "confusing", "corrupted", "crashing",
    "defective", "dirty", "disappointing", "disgusting", "dreadful", "dull",
    "faulty", "freezing", "frustrating", "garbage", "glitchy", "horrendous",
    "horrible", "inaccurate", "incompetent", "inferior", "irritating", "laggy",
    "lame", "lousy", "mediocre", "messy", "miserable", "noisy", "obnoxious",
    "outdated", "overpriced", "painful", "pathetic", "pointless", "poor",
    "ridiculous", "rude", "rubbish", "shoddy", "sloppy", "slow", "stale",
    "stupid", "subpar", "terrible", "trash", "unacceptable", "uncomfortable",
    "unhelpful", "unreliable", "unresponsive", "unusable", "useless", "weak",
    "worthless", "worst",
]

NEU_ADJS = [
    "acceptable", "adequate", "average", "basic", "common", "conventional",
    "decent", "fair", "fine", "functional", "generic", "moderate", "normal",
    "ordinary", "passable", "plain", "regular", "routine", "simple", "standard",
    "typical", "unchanged", "usable", "usual",
]

INTENSIFIERS = [
    "very", "really", "so", "extremely", "super", "truly", "quite",
    "honestly", "genuinely", "incredibly", "pretty", "totally",
]

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
ACTIONS = [
    "export", "upload", "download", "record", "transcribe", "summarize",
    "analyze", "configure", "install", "reset", "update", "save", "share", "open",
]
FORMATS = ["CSV", "PDF", "JSON", "MP3", "WAV", "TXT", "DOCX", "ZIP"]

POS_TEMPLATES = [
    # Domain 1: Conversational & Gratitude
    "Thank you so much, you have been {adv} {pos}!",
    "Thank you {adv} much, this {subj} is {pos}.",
    "Thanks a lot for your help, the {subj} worked like a charm.",
    "I {adv} appreciate your help, you were {adv2} {pos} today.",
    "You guys are {pos}, thank you for fixing the {subj} so quickly!",
    "Huge thanks to the {subj} team, everything is working {adv} well now.",
    "I am {adv} grateful for this {subj}, it saved me hours of work.",
    "Many thanks! That explanation was {adv} {pos} and easy to follow.",
    "You have been {adv} {pos} and patient with my questions, thank you.",
    "Props to the developers, this {subj} is an absolute lifesaver!",
    # Domain 2: Negation & Contrast (Positive)
    "Not bad at all, the {subj} is actually {adv} {pos}!",
    "I thought the {subj} would be {neg}, but it turned out {adv} {pos}.",
    "This {subj} never disappoints, it is always {pos} and {pos2}.",
    "I have zero complaints about the {subj}, everything works great.",
    "Could not be happier with the {subj}, it is {adv} {pos}.",
    "Nothing beats this {subj}, it is hands down the {pos} one.",
    "No issues whatsoever with the {subj}, setup was {pos} and {pos2}.",
    "It is not hard to use at all; in fact, the {subj} is {adv} {pos}.",
    # Domain 3: Tech, App & Voice AI Reviews
    "The {subj} is {adv} {pos} and processes everything in seconds.",
    "I love how {pos} and {pos2} the {subj} is after the latest patch.",
    "Voice recognition on this {subj} is {adv} {pos}, even with background noise.",
    "The {subj} {action}s files {adv} fast and the output is {pos}.",
    "Best {subj} I have used so far, the accuracy is {adv} {pos}.",
    "This {subj} exceeded my expectations, {adv} {pos} performance.",
    # Domain 4: General Reviews & Everyday Positive
    "The {subj} was {pos} and I would happily recommend it to everyone.",
    "{Adv} happy with the {subj}, everything felt {pos} and {pos2}.",
    "Such a {pos} {subj}, I can't stop smiling!",
    "I absolutely loved this {subj}, it was {adv} {pos}.",
    "The staff were {adv} {pos} and the whole {subj} experience was a joy.",
    "Everyone said the {subj} was {pos}, and they were completely right.",
    "What a {pos} {subj}! Totally worth the time and money.",
    "I'm thrilled with the {subj}; it felt {adv} {pos} from start to finish.",
    "The {subj} is {adv} {pos}, five stars from me without hesitation.",
    "I am {adv} glad I chose this {subj}, it is {pos} and {pos2}.",
    "Hands down the best {subj} on the market, {adv} {pos}!",
    "Everything about the {subj} is {pos}, I love using it every day.",
    # Domain 5: Short & Casual Inputs
    "{Adv} {pos} {subj}, love it!",
    "Super {pos} and {pos2}, works great!",
    "Awesome {subj}, {adv} {pos}!",
    "Great job on the {subj}, it is {adv} {pos}.",
    # Everyday wholesome & affirmative sentences
    "The sun is shining brightly today.",
    "I love spending time with my friends.",
    "She exercises every morning.",
    "We always eat dinner together.",
    "They take the bus to work.",
    "The weather is warm and sunny today.",
    "I enjoy spending quality time with my family.",
    "He goes for a morning walk in the park every day.",
    "We love cooking and eating meals together.",
    "She always smiles and greets everyone warmly.",
    "The flowers are blooming beautifully in the garden.",
    "The children are laughing and playing happily outside.",
    "Spending time with friends and family brings me joy.",
]

NEG_TEMPLATES = [
    # Domain 1: Conversational & Polite Complaints
    "Sorry, but this {subj} has not been {pos} at all, it is {adv} {neg}.",
    "No thank you, the {subj} was {adv} {neg} and {neg2}.",
    "I hate to complain, but the {subj} is {adv} {neg} right now.",
    "Please fix the {subj}, it keeps failing and is {adv} {neg} to use.",
    "I asked for help hours ago and the {subj} was {adv} {neg}.",
    "Unfortunately, you have not been {pos} and my {subj} is still broken.",
    "I am {adv} disappointed with your {subj}, it wasted my whole afternoon.",
    "Why is the {subj} so {neg}? It barely works anymore.",
    # Domain 2: Negation & Contrast (Negative)
    "Honestly, the {subj} was not {pos} at all, it felt {adv} {neg}.",
    "I would not call this {subj} {pos}; it is actually {adv} {neg}.",
    "The {subj} is not {pos}, not {pos2}, and definitely not worth it.",
    "It used to be {pos}, but after the update the {subj} is {adv} {neg}.",
    "Not happy with the {subj} at all, it is {neg} and {neg2}.",
    "Never buying or using this {subj} again, what {neg} quality.",
    "I can't recommend this {subj} because it is neither {pos} nor reliable.",
    "Hardly works as advertised—the {subj} is {adv} {neg}.",
    # Domain 3: Tech, App & Voice AI Issues
    "The {subj} keeps crashing whenever I try to {action} a {fmt} file.",
    "Audio transcription in the {subj} is {adv} {neg} and misses half the words.",
    "This {subj} is {adv} {neg}, it freezes every time I click analyze.",
    "The {subj} failed to meet my expectations, {adv} {neg} and full of bugs.",
    # Domain 4: General Reviews & Everyday Negative
    "The {subj} was {neg} and I would not recommend it to anyone.",
    "{Adv} unhappy with the {subj}, everything felt {neg} and {neg2}.",
    "Such a {neg} {subj}, I am {adv} annoyed and frustrated.",
    "I {adv} hated this {subj}, it was {adv2} {neg}.",
    "Worst {subj} I have ever tried, {adv} {neg}.",
    "The staff were {adv} {neg} and the whole {subj} experience was a pain.",
    "Everyone said the {subj} was {neg}, and sadly they were right.",
    "What a {neg} {subj}! Complete waste of time and money.",
    "I'm furious about the {subj}; it felt {adv} {neg} from the start.",
    "The {subj} is {adv} {neg}, one star from me.",
    "I deeply regret choosing this {subj}, it is {adv} {neg} and {neg2}.",
    "Stay away from this {subj}, it is completely {neg}.",
    # Domain 5: Short & Casual Inputs
    "{Adv} {neg} {subj}, total waste.",
    "Super {neg} and {neg2}, do not use.",
    "Terrible {subj}, {adv} {neg}!",
    "Broken and {neg}, the {subj} does not work.",
]

NEU_TEMPLATES = [
    # Domain 1: Questions & Help Queries
    "How do I {action} the {subj} output as a {fmt} file?",
    "Can you tell me how the {subj} works and when it opens on {day}?",
    "Where can I find the settings to {action} the {subj}?",
    "Does the {subj} support {fmt} format or only standard text?",
    "What time does the {subj} maintenance start on {day}?",
    "Is there a shortcut to {action} data in the {subj}?",
    "Who should I contact regarding the {subj} documentation?",
    # Domain 2: Mixed / Balanced / Average Statements
    "The {subj} is {neu}—nothing special, just does the basic job.",
    "The {subj} works okay for simple tasks, though it is fairly {neu}.",
    "It is an {neu} {subj}, neither particularly good nor bad.",
    "So far the {subj} seems {neu}, I will test it more on {day}.",
    "The {subj} has a {neu} layout and standard features.",
    "Not sure how I feel about the {subj} yet, it is pretty {neu}.",
    # Domain 3, 4, 5: Factual, Technical & Operational Statements
    "There are {n} versions of the {subj} available on the website.",
    "The {subj} starts at {n} pm and ends at {n2} pm on {day}.",
    "The {subj} was moved to the second floor last {day}.",
    "According to the manual, the {subj} needs {n} steps to {action}.",
    "The {subj} is scheduled for routine maintenance on {day} in {month}.",
    "I bought the {subj} from the store on {day} for {n} dollars.",
    "The {subj} costs {n} dollars and ships within {n2} business days.",
    "The {subj} has {n} main modules and exports to {fmt}.",
    "She mentioned the {subj} during the team call on {day}.",
    "The report describes how the {subj} processes {fmt} files.",
    "The {subj} comes in {n} different configurations.",
    "The meeting about the {subj} is scheduled for {day} at noon.",
    "A new version of the {subj} was announced by the company in {month}.",
    "The package contains the {subj}, a manual, and {n} cables.",
    "Tomorrow we will discuss the {subj} requirements in the office.",
    "The {subj} was released in {month} and updated on {day}.",
    "I have been using the {subj} for {n} days to {action} {fmt} files.",
    "Please {action} the {subj} before logging out of the system.",
    "Users can {action} up to {n} items at a time using the {subj}.",
]

STANDALONE_POS = [
    "Thank you!", "Thanks so much!", "Thank you very much!", "Thanks a lot!",
    "You have been very helpful.", "Thank you so much, you have been very helpful.",
    "Super helpful, thank you!", "Really appreciate your help!", "Great work!",
    "Awesome job!", "Love it!", "Works like a charm!", "Spot on, thanks!",
    "Brilliant, thank you so much!", "Very nice and easy to use.",
    "Not bad at all, actually pretty good!", "Five stars, highly recommend!",
    "Everything works perfectly now, thanks!", "So glad this worked, thank you!",
    "Absolutely fantastic experience!", "Best tool ever!", "Very fast and accurate.",
]

STANDALONE_NEG = [
    "Terrible experience.", "Does not work at all.", "Complete waste of time.",
    "Very disappointed.", "So frustrating and slow!", "Total garbage, do not use.",
    "It keeps crashing every single time.", "Worst experience ever.",
    "Not helpful at all.", "Very unhelpful and rude.", "I hate this, it is broken.",
    "Nothing works as expected.", "Horrible accuracy, completely wrong.",
    "Waste of money and time.", "Full of bugs and super laggy.",
    "Not good at all, very bad.", "I regret using this.", "Unacceptable quality.",
]

STANDALONE_NEU = [
    "How does this work?", "Testing the microphone one two three.",
    "Hello, is anyone there?", "Let me check the documentation first.",
    "It is okay, nothing special.", "Just average, does what it says.",
    "The file was uploaded five minutes ago.", "Where is the export button?",
    "Let's schedule the meeting for tomorrow at 3 pm.", "Click analyze to see the results.",
    "The system is currently running version two.", "No change from the previous version.",
    "Can you explain how to use this feature?", "Just a normal test sentence.",
]


def _fill(template: str, rng: random.Random) -> str:
    adv = rng.choice(INTENSIFIERS)
    act = rng.choice(ACTIONS)
    return template.format(
        subj=rng.choice(SUBJECTS),
        pos=rng.choice(POS_ADJS),
        pos2=rng.choice(POS_ADJS),
        neg=rng.choice(NEG_ADJS),
        neg2=rng.choice(NEG_ADJS),
        neu=rng.choice(NEU_ADJS),
        adv=adv,
        Adv=adv.capitalize(),
        adv2=rng.choice(INTENSIFIERS),
        day=rng.choice(DAYS),
        month=rng.choice(MONTHS),
        action=act,
        Action=act.capitalize(),
        fmt=rng.choice(FORMATS),
        n=rng.randint(2, 15),
        n2=rng.randint(2, 15),
    )


def build(path: str | Path = SENTIMENT_DATASET_PATH, per_class: int = 1000) -> None:
    rng = random.Random(42)
    rows: list[tuple[str, str]] = []

    for label, templates, standalones in [
        ("Positive", POS_TEMPLATES, STANDALONE_POS),
        ("Negative", NEG_TEMPLATES, STANDALONE_NEG),
        ("Neutral", NEU_TEMPLATES, STANDALONE_NEU),
    ]:
        seen: set[str] = set(standalones)
        for s in standalones:
            rows.append((s, label))
        while len(seen) < per_class:
            sentence = _fill(rng.choice(templates), rng)
            if sentence not in seen:
                seen.add(sentence)
                rows.append((sentence, label))

    rng.shuffle(rows)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        os.chmod(out, stat.S_IWRITE)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(rows)


if __name__ == "__main__":
    build(SENTIMENT_DATASET_PATH)
    print(f"Rebuilt {SENTIMENT_DATASET_PATH} successfully with 3,000 samples across 5 domains.")