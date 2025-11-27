import pandas as pd
import os
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ... (imports remain the same)

# 1. Load Data
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, 'dataset.csv')

try:
    df = pd.read_csv(csv_path)
    print("Data loaded successfully!")
    
    # --- DEBUGGING STEP ---
    # Strip spaces from column names (fixes "Comments " vs "Comments")
    df.columns = df.columns.str.strip()
    
    print(f"Columns found in CSV: {df.columns.tolist()}")
    
    # Check if 'Comments' exists, if not, try to find a likely match
    if 'Comments' not in df.columns:
        # Check for lowercase 'comments'
        if 'comments' in df.columns:
            print("Found 'comments' (lowercase). Renaming to 'Comments'...")
            df.rename(columns={'comments': 'Comments'}, inplace=True)
        # Check for 'text'
        elif 'text' in df.columns:
             print("Found 'text'. Renaming to 'Comments'...")
             df.rename(columns={'text': 'Comments'}, inplace=True)
        else:
            # If we still can't find it, we stop and ask you to look at the output
            print("\nCRITICAL ERROR: Could not find a 'Comments' column.")
            print("Please update the script to use one of the column names listed above.")
            exit()
            
    if 'Label' not in df.columns:
        # Handle label column variations too
        if 'label' in df.columns:
             df.rename(columns={'label': 'Label'}, inplace=True)

except FileNotFoundError:
    print("File not found.")
    exit()

# ... (Rest of your script: cleaning, training, etc.)

# 2. Preprocessing function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    return text

print("Cleaning data...")
df['clean_text'] = df['Comments'].apply(clean_text)

# 3. Split Data
X = df['clean_text']
y = df['Label'] # 0 for non-hate, 1 for hate

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Vectorization (Convert text to numbers)
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# --- ALGORITHM 1: Naive Bayes ---
print("Training Naive Bayes...")
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)
print(f"Naive Bayes Accuracy: {accuracy_score(y_test, nb_model.predict(X_test_vec))}")

# --- ALGORITHM 2: Logistic Regression ---
print("Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_vec, y_train)
print(f"LR Accuracy: {accuracy_score(y_test, lr_model.predict(X_test_vec))}")

# 5. Save the Models and Vectorizer
print("Saving models...")
# We save the vectorizer once (it's shared)
joblib.dump(vectorizer, 'vectorizer.pkl')

# Save the models separately
joblib.dump(nb_model, 'model_nb.pkl')
joblib.dump(lr_model, 'model_lr.pkl')

print("Training complete. Files saved.")