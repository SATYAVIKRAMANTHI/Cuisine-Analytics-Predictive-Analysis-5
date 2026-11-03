# Cuisine-Analytics-Predictive-Analysis-5
Restaurant Data Analysis and Cuisine Classification using Python and Machine Learning.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA

from mlxtend.frequent_patterns import apriori, association_rules

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

print("=" * 60)
print("CUISINE ANALYTICS - PREDICTIVE ANALYSIS")
print("=" * 60)

df = pd.read_csv("/Users/satyavikramanthi/Library/Containers/net.whatsapp.WhatsApp/Data/tmp/documents/EFCEB09E-A190-4732-B544-6FC359F7076E/swiggy_file.csv")

print("\nDATASET LOADED SUCCESSFULLY")
print("Dataset Shape:", df.shape)
print(df.head())

df.columns = df.columns.str.strip()

required_columns = [
    "Restaurant Name", "Cuisine", "Rating",
    "Number of Ratings", "Average Price",
    "Number of Offers", "Offer Name",
    "Area", "Pure Veg", "Location"
]

missing_columns = [c for c in required_columns if c not in df.columns]

if missing_columns:
    raise ValueError("Missing columns: " + str(missing_columns))

df = df.drop_duplicates().copy()

df["Rating"] = pd.to_numeric(
    df["Rating"].astype(str).str.extract(r"(\d+\.?\d*)")[0],
    errors="coerce"
)

df["Average Price"] = pd.to_numeric(
    df["Average Price"].astype(str).str.replace(",", "", regex=False).str.extract(r"(\d+\.?\d*)")[0],
    errors="coerce"
)

rating_count = df["Number of Ratings"].astype(str).str.upper()

rating_count = rating_count.str.replace(",", "", regex=False)

def convert_rating_count(value):
    value = str(value).upper().replace(",", "").strip()

    match = pd.Series([value]).str.extract(r"(\d+\.?\d*)")[0].iloc[0]

    if pd.isna(match):
        return np.nan

    number = float(match)

    if "K" in value:
        number *= 1000
    elif "M" in value:
        number *= 1000000

    return number

df["Number of Ratings"] = df["Number of Ratings"].apply(convert_rating_count)

df["Number of Offers"] = pd.to_numeric(
    df["Number of Offers"], errors="coerce"
)

for col in df.select_dtypes(include=np.number).columns:
    df[col] = df[col].fillna(df[col].median())

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].fillna("Unknown")

df["Cuisine"] = df["Cuisine"].astype(str).str.strip()
df["Primary Cuisine"] = df["Cuisine"].apply(
    lambda x: x.split(",")[0].strip()
)

df["Pure Veg"] = df["Pure Veg"].astype(str).str.strip().str.title()

print("\nDATA CLEANING COMPLETED")
print(df.isnull().sum())
print(df.dtypes)

print("\nDATASET DESCRIPTION")
print(df.describe())

print("\n" + "=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

plt.figure(figsize=(9, 5))
sns.histplot(df["Rating"], kde=True)
plt.title("Restaurant Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 5))
sns.histplot(df["Average Price"], kde=True)
plt.title("Average Restaurant Price Distribution")
plt.xlabel("Average Price")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
df["Primary Cuisine"].value_counts().head(10).plot(kind="bar")
plt.title("Top 10 Popular Cuisines")
plt.xlabel("Cuisine")
plt.ylabel("Restaurant Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
df["Location"].value_counts().head(10).plot(kind="bar")
plt.title("Top 10 Restaurant Locations")
plt.xlabel("Location")
plt.ylabel("Restaurant Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 6))
sns.heatmap(
    df[["Rating", "Number of Ratings", "Average Price", "Number of Offers"]].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 5))
sns.scatterplot(
    data=df,
    x="Average Price",
    y="Rating"
)
plt.title("Average Price vs Restaurant Rating")
plt.tight_layout()
plt.show()

top_cuisines = df["Primary Cuisine"].value_counts().head(10).index

plt.figure(figsize=(11, 6))
sns.boxplot(
    data=df[df["Primary Cuisine"].isin(top_cuisines)],
    x="Primary Cuisine",
    y="Rating"
)
plt.title("Cuisine-wise Rating Comparison")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\nEDA COMPLETED SUCCESSFULLY")

print("\n" + "=" * 60)
print("STATISTICAL HYPOTHESIS TESTING")
print("=" * 60)

rating_data = df["Rating"].dropna()

t_stat, p_value = stats.ttest_1samp(
    rating_data,
    popmean=3.5
)

print("\nONE SAMPLE T-TEST")
print("Null Hypothesis: Average restaurant rating is 3.5")
print("T Statistic:", t_stat)
print("P Value:", p_value)

if p_value < 0.05:
    print("Result: Reject the Null Hypothesis")
else:
    print("Result: Fail to Reject the Null Hypothesis")

groups = [
    group["Rating"].values
    for _, group in df.groupby("Primary Cuisine")
    if len(group) >= 2
]

if len(groups) >= 2:
    f_stat, anova_p = stats.f_oneway(*groups)

    print("\nANOVA TEST")
    print("F Statistic:", f_stat)
    print("P Value:", anova_p)

    if anova_p < 0.05:
        print("Significant rating differences exist between cuisines")
    else:
        print("No statistically significant rating differences found")

print("\n" + "=" * 60)
print("LINEAR AND POLYNOMIAL REGRESSION")
print("=" * 60)

reg_data = df[["Average Price", "Rating"]].dropna()

X_reg = reg_data[["Average Price"]]
y_reg = reg_data["Rating"]

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg,
    y_reg,
    test_size=0.2,
    random_state=42
)

linear_model = LinearRegression()

linear_model.fit(X_train_reg, y_train_reg)

linear_predictions = linear_model.predict(X_test_reg)

print("\nLINEAR REGRESSION")
print("R2 Score:", r2_score(y_test_reg, linear_predictions))
print("Mean Squared Error:", mean_squared_error(y_test_reg, linear_predictions))

poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train_reg)
X_test_poly = poly.transform(X_test_reg)

poly_model = LinearRegression()

poly_model.fit(X_train_poly, y_train_reg)

poly_predictions = poly_model.predict(X_test_poly)

print("\nPOLYNOMIAL REGRESSION")
print("R2 Score:", r2_score(y_test_reg, poly_predictions))
print("Mean Squared Error:", mean_squared_error(y_test_reg, poly_predictions))

plt.figure(figsize=(9, 5))
plt.scatter(X_test_reg, y_test_reg, label="Actual")
plt.scatter(X_test_reg, linear_predictions, label="Predicted")
plt.title("Linear Regression: Price vs Rating")
plt.xlabel("Average Price")
plt.ylabel("Rating")
plt.legend()
plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("CUISINE CLASSIFICATION")
print("=" * 60)

target_counts = df["Primary Cuisine"].value_counts()

common_cuisines = target_counts[target_counts >= 5].index

classification_df = df[
    df["Primary Cuisine"].isin(common_cuisines)
].copy()

classification_df["Target Cuisine"] = classification_df["Primary Cuisine"]

X = classification_df[
    [
        "Rating",
        "Number of Ratings",
        "Average Price",
        "Number of Offers",
        "Pure Veg",
        "Location",
        "Area"
    ]
]

y = classification_df["Target Cuisine"]

if y.nunique() < 2:
    raise ValueError("Not enough cuisine categories for classification.")

numeric_features = [
    "Rating",
    "Number of Ratings",
    "Average Price",
    "Number of Offers"
]

categorical_features = [
    "Pure Veg",
    "Location",
    "Area"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numeric_features
        ),
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical_features
        )
    ]
)

stratify_value = y if y.value_counts().min() >= 2 else None

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=stratify_value
)

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Neural Network": MLPClassifier(
        hidden_layer_sizes=(50, 25),
        max_iter=500,
        random_state=42
    )
}

results = {}
trained_models = {}

for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    results[name] = accuracy
    trained_models[name] = pipeline

    print("\n", name)
    print("Accuracy:", round(accuracy * 100, 2), "%")

best_model_name = max(results, key=results.get)

best_model = trained_models[best_model_name]

print("\nBEST CLASSIFICATION MODEL:", best_model_name)
print("Best Accuracy:", round(results[best_model_name] * 100, 2), "%")

best_predictions = best_model.predict(X_test)

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test,
    best_predictions,
    zero_division=0
))

plt.figure(figsize=(10, 7))

labels = best_model.named_steps["classifier"].classes_

sns.heatmap(
    confusion_matrix(y_test, best_predictions, labels=labels),
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)

plt.title("Cuisine Classification Confusion Matrix")
plt.xlabel("Predicted Cuisine")
plt.ylabel("Actual Cuisine")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
pd.Series(results).mul(100).plot(kind="bar")
plt.title("Comparison of Machine Learning Models")
plt.ylabel("Accuracy (%)")
plt.xlabel("ML Models")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("RANDOM FOREST AND GRADIENT BOOSTING")
print("=" * 60)

rf_model = trained_models["Random Forest"]
gb_model = trained_models["Gradient Boosting"]

print("Random Forest Accuracy:",
      round(results["Random Forest"] * 100, 2), "%")

print("Gradient Boosting Accuracy:",
      round(results["Gradient Boosting"] * 100, 2), "%")

print("\n" + "=" * 60)
print("K-MEANS CLUSTERING")
print("=" * 60)

cluster_data = df[
    [
        "Rating",
        "Number of Ratings",
        "Average Price",
        "Number of Offers"
    ]
].copy()

cluster_scaler = StandardScaler()

scaled_cluster_data = cluster_scaler.fit_transform(cluster_data)

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["KMeans Cluster"] = kmeans.fit_predict(scaled_cluster_data)

print(df[["Restaurant Name", "KMeans Cluster"]].head(10))

plt.figure(figsize=(9, 5))
sns.scatterplot(
    x=scaled_cluster_data[:, 0],
    y=scaled_cluster_data[:, 2],
    hue=df["KMeans Cluster"],
    palette="viridis"
)
plt.title("K-Means Restaurant Clustering")
plt.xlabel("Standardized Rating")
plt.ylabel("Standardized Average Price")
plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("HIERARCHICAL CLUSTERING")
print("=" * 60)

sample_size = min(500, len(scaled_cluster_data))

hierarchical = AgglomerativeClustering(n_clusters=3)

hierarchical_labels = hierarchical.fit_predict(
    scaled_cluster_data[:sample_size]
)

df["Hierarchical Cluster"] = -1

df.loc[
    df.index[:sample_size],
    "Hierarchical Cluster"
] = hierarchical_labels

print(df[["Restaurant Name", "Hierarchical Cluster"]].head(10))

plt.figure(figsize=(9, 5))
plt.scatter(
    scaled_cluster_data[:sample_size, 0],
    scaled_cluster_data[:sample_size, 2],
    c=hierarchical_labels
)
plt.title("Hierarchical Clustering")
plt.xlabel("Standardized Rating")
plt.ylabel("Standardized Average Price")
plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("ASSOCIATION RULE MINING")
print("=" * 60)

transactions = pd.DataFrame(index=df.index)

for cuisine in df["Primary Cuisine"].value_counts().head(15).index:
    transactions[cuisine] = (
        df["Cuisine"].str.contains(
            cuisine,
            case=False,
            regex=False
        )
    )

transactions["Pure Veg"] = df["Pure Veg"].str.lower().eq("yes")

transactions["Offers Available"] = df["Number of Offers"] > 0

transactions = transactions.astype(bool)

frequent_items = apriori(
    transactions,
    min_support=0.03,
    use_colnames=True
)

if not frequent_items.empty:
    rules = association_rules(
        frequent_items,
        metric="confidence",
        min_threshold=0.3
    )

    if not rules.empty:
        print(rules[
            ["antecedents", "consequents", "support", "confidence", "lift"]
        ].sort_values("lift", ascending=False).head(10))
    else:
        print("No strong association rules found.")
else:
    print("No frequent itemsets found.")

print("\n" + "=" * 60)
print("PRINCIPAL COMPONENT ANALYSIS")
print("=" * 60)

pca = PCA(n_components=2)

pca_results = pca.fit_transform(scaled_cluster_data)

print("Explained Variance Ratio:", pca.explained_variance_ratio_)
print("Total Explained Variance:", sum(pca.explained_variance_ratio_))

plt.figure(figsize=(9, 5))
plt.scatter(
    pca_results[:, 0],
    pca_results[:, 1],
    c=df["KMeans Cluster"],
    cmap="viridis"
)
plt.title("Principal Component Analysis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("RESTAURANT CUISINE PREDICTION")
print("=" * 60)

print("\nEnter Restaurant Details")

try:
    user_rating = float(input("Enter Restaurant Rating (0-5): "))
    user_price = float(input("Enter Average Price: "))
    user_ratings_count = float(input("Enter Number of Ratings: "))
    user_offers = float(input("Enter Number of Offers: "))
    user_veg = input("Is it Pure Veg? (Yes/No): ").strip().title()
    user_location = input("Enter Location: ").strip()
    user_area = input("Enter Area: ").strip()

    if not 0 <= user_rating <= 5:
        raise ValueError("Rating must be between 0 and 5.")

    if min(user_price, user_ratings_count, user_offers) < 0:
        raise ValueError("Price, ratings count and offers cannot be negative.")

    user_input = pd.DataFrame({
        "Rating": [user_rating],
        "Number of Ratings": [user_ratings_count],
        "Average Price": [user_price],
        "Number of Offers": [user_offers],
        "Pure Veg": [user_veg],
        "Location": [user_location],
        "Area": [user_area]
    })

    predicted_cuisine = best_model.predict(user_input)

    print("\nPREDICTED CUISINE:", predicted_cuisine[0])

except ValueError as e:
    print("Invalid Input:", e)

print("\n" + "=" * 60)
print("CUISINE ANALYTICS PROJECT COMPLETED")
print("=" * 60)

df.to_csv("Cuisine_Analytics_Results.csv", index=False)

print("Results saved to Cuisine_Analytics_Results.csv")
