# ================================================================
# URL ABNORMAL PATTERN DETECTION USING FEATURE ENGINEERING
# ================================================================

# -----------------------------
# 1. IMPORT LIBRARIES
# -----------------------------

import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ================================================================
# 2. CREATE DATASET
# ================================================================

data = {
    "url": [
        # ---------------- NORMAL URLs ----------------
        "https://www.google.com",
        "https://www.microsoft.com",
        "https://www.amazon.com",
        "https://www.wikipedia.org",
        "https://www.github.com",
        "https://www.apple.com",
        "https://www.linkedin.com",
        "https://www.python.org",
        "https://www.ibm.com",
        "https://www.coursera.org",
        "https://www.youtube.com",
        "https://www.facebook.com",
        "https://www.netflix.com",
        "https://www.reddit.com",
        "https://www.adobe.com",

        # ---------------- ABNORMAL URLs ----------------
        "http://192.168.1.10/login",
        "http://secure-login-account.com/verify",
        "http://google-login-security.com/account/verify",
        "http://bank-account-verification.com/login",
        "http://free-gift-card.com/claim-now",
        "http://paypal-security-login.com/verify/account",
        "http://login.verify.account-security.com/update",
        "http://www.example.com/@login@verify",
        "http://192.168.10.20/secure/login/account",
        "http://secure-bank-login.com/verify/user/account/password",
        "http://account-security-verification-login.com/update",
        "http://free-money-gift-card-login.com/claim",
        "http://verify-your-bank-account-login.com/password",
        "http://secure-login-confirm-account.com/verify",
        "http://192.168.100.25/login/verify/account"
    ],

    # 0 = Normal
    # 1 = Abnormal
    "label": [
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,

        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1
    ]
}

df = pd.DataFrame(data)

print("=" * 70)
print("URL ABNORMAL PATTERN DETECTION")
print("=" * 70)

print("\nDataset:")
print(df.to_string(index=False))


# ================================================================
# 3. URL FEATURE EXTRACTION FUNCTIONS
# ================================================================

def has_ip_address(url):
    """
    Check whether the URL contains an IPv4 address.
    """

    pattern = r'(\d{1,3}\.){3}\d{1,3}'

    if re.search(pattern, url):
        return 1

    return 0


def count_special_characters(url):
    """
    Count special characters commonly found in suspicious URLs.
    """

    special_characters = r'[@?=&%_\-]'

    return len(re.findall(special_characters, url))


def count_suspicious_keywords(url):
    """
    Count suspicious keywords present in the URL.
    """

    suspicious_keywords = [
        "login",
        "signin",
        "sign-in",
        "verify",
        "verification",
        "account",
        "secure",
        "security",
        "bank",
        "password",
        "update",
        "confirm",
        "confirmation",
        "free",
        "gift",
        "claim"
    ]

    url_lower = url.lower()

    count = 0

    for word in suspicious_keywords:
        if word in url_lower:
            count += 1

    return count


def extract_features(url):
    """
    Extract numerical features from a URL.
    """

    # Parse URL
    parsed_url = urlparse(url)

    # Extract hostname
    hostname = parsed_url.netloc

    # Remove username information if present
    if "@" in hostname:
        hostname = hostname.split("@")[-1]

    # Remove port number
    hostname = hostname.split(":")[0]

    # Dictionary for storing features
    features = {}

    # ------------------------------------------------
    # Feature 1: URL Length
    # ------------------------------------------------

    features["url_length"] = len(url)

    # ------------------------------------------------
    # Feature 2: Hostname Length
    # ------------------------------------------------

    features["hostname_length"] = len(hostname)

    # ------------------------------------------------
    # Feature 3: Number of Dots
    # ------------------------------------------------

    features["dot_count"] = url.count(".")

    # ------------------------------------------------
    # Feature 4: Number of Hyphens
    # ------------------------------------------------

    features["hyphen_count"] = url.count("-")

    # ------------------------------------------------
    # Feature 5: Number of Slashes
    # ------------------------------------------------

    features["slash_count"] = url.count("/")

    # ------------------------------------------------
    # Feature 6: Number of Digits
    # ------------------------------------------------

    features["digit_count"] = sum(
        character.isdigit()
        for character in url
    )

    # ------------------------------------------------
    # Feature 7: Number of Special Characters
    # ------------------------------------------------

    features["special_char_count"] = count_special_characters(url)

    # ------------------------------------------------
    # Feature 8: Number of Subdomains
    # ------------------------------------------------

    if hostname:

        hostname_parts = hostname.split(".")

        if len(hostname_parts) > 2:
            features["subdomain_count"] = len(hostname_parts) - 2
        else:
            features["subdomain_count"] = 0

    else:

        features["subdomain_count"] = 0

    # ------------------------------------------------
    # Feature 9: IP Address
    # ------------------------------------------------

    features["has_ip"] = has_ip_address(url)

    # ------------------------------------------------
    # Feature 10: @ Symbol
    # ------------------------------------------------

    if "@" in url:
        features["has_at_symbol"] = 1
    else:
        features["has_at_symbol"] = 0

    # ------------------------------------------------
    # Feature 11: HTTPS
    # ------------------------------------------------

    if parsed_url.scheme.lower() == "https":
        features["is_https"] = 1
    else:
        features["is_https"] = 0

    # ------------------------------------------------
    # Feature 12: Path Length
    # ------------------------------------------------

    features["path_length"] = len(parsed_url.path)

    # ------------------------------------------------
    # Feature 13: Query Length
    # ------------------------------------------------

    features["query_length"] = len(parsed_url.query)

    # ------------------------------------------------
    # Feature 14: Suspicious Keyword Count
    # ------------------------------------------------

    features["suspicious_word_count"] = (
        count_suspicious_keywords(url)
    )

    return features


# ================================================================
# 4. EXTRACT FEATURES FROM ALL URLs
# ================================================================

feature_list = []

for url in df["url"]:

    features = extract_features(url)

    feature_list.append(features)


# Convert extracted features into DataFrame
features_df = pd.DataFrame(feature_list)


print("\n" + "=" * 70)
print("EXTRACTED FEATURES")
print("=" * 70)

print(features_df.to_string(index=False))


# ================================================================
# 5. COMBINE URL DATA WITH FEATURES
# ================================================================

final_df = pd.concat(
    [df, features_df],
    axis=1
)

print("\n" + "=" * 70)
print("FINAL FEATURE DATASET")
print("=" * 70)

print(final_df.to_string(index=False))


# ================================================================
# 6. PREPARE INPUT AND OUTPUT
# ================================================================

# X = Features
X = features_df

# y = Target/Label
y = df["label"]


# ================================================================
# 7. SPLIT DATA INTO TRAINING AND TESTING
# ================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


print("\n" + "=" * 70)
print("DATASET SPLIT")
print("=" * 70)

print("Total samples     :", len(X))
print("Training samples  :", len(X_train))
print("Testing samples   :", len(X_test))


# ================================================================
# 8. CREATE RANDOM FOREST MODEL
# ================================================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)


# ================================================================
# 9. TRAIN MODEL
# ================================================================

model.fit(
    X_train,
    y_train
)

print("\nModel training completed successfully.")

# 10. MAKE PREDICTIONS
y_pred = model.predict(X_test)

# 11. CALCULATE ACCURACY
accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)


# ================================================================
# 12. CLASSIFICATION REPORT
# ================================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=[
            "Normal",
            "Abnormal"
        ],
        zero_division=0
    )
)


# 13. CONFUSION MATRIX

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=[
        "Normal",
        "Abnormal"
    ],
    yticklabels=[
        "Normal",
        "Abnormal"
    ]
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Confusion Matrix - URL Detection")

plt.tight_layout()

plt.show()


 # 14. FEATURE IMPORTANCE


importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

print(
    importance_df.to_string(index=False)
)

# 15. FEATURE IMPORTANCE GRAPH

plt.figure(figsize=(10, 6))

plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance for URL Abnormal Pattern Detection")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.show()

# 16. FUNCTION TO PREDICT A NEW URL

def predict_url(url):

    # Extract features
    features = extract_features(url)

    # Convert to DataFrame
    feature_input = pd.DataFrame(
        [features]
    )

    # Make prediction
    prediction = model.predict(
        feature_input
    )[0]

    # Get probability
    probability = model.predict_proba(
        feature_input
    )[0]

    abnormal_probability = probability[1] * 100

    print("\n" + "-" * 70)
    print("URL ANALYSIS")
    print("-" * 70)

    print("URL:")
    print(url)

    print("\nExtracted Features:")

    for feature, value in features.items():
        print(
            f"{feature:<25}: {value}"
        )

    print("\nPrediction:")

    if prediction == 0:

        print("NORMAL")

    else:

        print("ABNORMAL / SUSPICIOUS")

    print(
        f"Abnormal Probability: "
        f"{abnormal_probability:.2f}%"
    )

    print("-" * 70)

# 17. TEST NEW URLS
test_urls = [

    "https://www.google.com",

    "https://www.github.com",

    "http://192.168.1.50/login",

    "http://secure-bank-login.com/verify/password",

    "http://example.com"

]


print("\n" + "=" * 70)
print("TESTING NEW URLs")
print("=" * 70)


for url in test_urls:

    predict_url(url)

# 18. END

print("\n" + "=" * 70)
print("PROGRAM EXECUTION COMPLETED")
print("=" * 70)