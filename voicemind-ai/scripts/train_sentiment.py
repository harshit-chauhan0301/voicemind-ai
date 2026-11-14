"""Train the sentiment model:  python -m scripts.train_sentiment"""
from core.sentiment import train_and_save

if __name__ == "__main__":
    m = train_and_save()
    print(f"Trained on {m['train_size']} samples, tested on {m['test_size']}.\n")
    print(f"{'Model':<22}{'Accuracy':>10}{'F1 (macro)':>13}")
    for row in m["comparison"]:
        print(f"{row['model']:<22}{row['accuracy']:>10}{row['f1_macro']:>13}")
    print("\nConfusion matrix (rows = true, cols = predicted):", m["labels"])
    for label, row in zip(m["labels"], m["confusion_matrix"]):
        print(f"  {label:<9}{row}")
    print("\nSaved the Logistic Regression model to models/sentiment_model.joblib")
