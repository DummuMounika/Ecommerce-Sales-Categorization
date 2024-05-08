import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.utils import shuffle
import string
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class TextClassifier:
    def __init__(self):
        # Initialize CountVectorizer, MultinomialNB, and WordNet Lemmatizer
        self.vectorizer = CountVectorizer(max_features=5000, ngram_range=(1, 2))
        self.classifier = MultinomialNB()
        self.lemmatizer = WordNetLemmatizer()
        # Read invalid words from file
        self.invalid_words = self._read_invalid_words('Input/Ignore_words.txt')

    def _preprocess_text(self, text):
        # Preprocess text data
        if pd.isnull(text) or not isinstance(text, str):
            return None
        else:
            text = ''.join(char for char in text if char.isalnum() or char.isspace())
            if text.isdigit() or all(char in string.punctuation for char in text):
                return None
            tokens = word_tokenize(text.lower())
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
            return " ".join(tokens)

    def _read_invalid_words(self, file_path):
        # Read invalid words from file
        with open(file_path, 'r') as file:
            invalid_words = file.read().splitlines()
        return invalid_words

    def contains_invalid_words(self, description):
        # Check if description contains invalid words
        for word in self.invalid_words:
            if word.lower() in description.lower():
                return True
        return False

    def train(self, df):
        # Train the classifier
        df = shuffle(df)
        df['processed_text'] = df['Description'].apply(self._preprocess_text)
        df = df.dropna(subset=['processed_text'])
        valid_records = df[~df['Description'].apply(self.contains_invalid_words)]
        X_vectorized = self.vectorizer.fit_transform(valid_records['processed_text'])
        X_train, X_test, y_train, y_test = train_test_split(X_vectorized, valid_records['Category'], test_size=0.2, random_state=42)
        self.classifier.fit(X_train, y_train)
        predictions = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, predictions) * 100
        return accuracy

    def predict(self, new_df):
        # Predict categories for new descriptions
        new_df['processed_text'] = new_df['Description'].apply(self._preprocess_text)
        new_df = new_df.dropna(subset=['processed_text'])
        valid_new_df = new_df[~new_df['Description'].apply(self.contains_invalid_words)]
        X_vectorized_new = self.vectorizer.transform(valid_new_df['processed_text'])
        predicted_categories = self.classifier.predict(X_vectorized_new)
        valid_new_df = valid_new_df.copy()
        valid_new_df.loc[:, 'Category'] = predicted_categories
        return valid_new_df

    def save_predictions(self, predicted_df, file_name_csv='Output/predicted_categories.csv', file_name_json='Output/predicted_categories.json'):
        # Save predictions to CSV and JSON files
        predicted_df.to_csv(file_name_csv, index=False)
        predicted_categories_dict = predicted_df.to_dict(orient='records')
        with open(file_name_json, 'w') as json_file:
            json.dump(predicted_categories_dict, json_file)

    def plot_category_distribution(self, df):
        # Plot distribution of categories
        category_counts = df['Category'].value_counts()
        majority_category = category_counts.idxmax()
        plt.figure(figsize=(10, 6))
        category_counts.plot(kind='bar', color='skyblue')
        plt.title('Distribution of Categories')
        plt.xlabel('Category')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        # Add labels to the bars
        for i, sales in enumerate(category_counts):
            plt.text(i, sales, f'{int(sales):,}', ha='center', va='bottom')
        plt.axhline(y=category_counts[majority_category], color='r', linestyle='--', label='Majority Category')
        plt.legend()
        plt.tight_layout()
        plt.savefig('Output/highest_category.png')
        plt.show()
        print("Majority Category:", majority_category)
        print("Number of occurrences:", category_counts[majority_category])

    def Analyze_top_customers(self, df):
        # Plot top customers by total quantity
        customer_quantity = df.groupby('CustomerID')['Quantity'].sum().sort_values(ascending=False)
        # Convert the customer ID to integer format
        customer_quantity.index = customer_quantity.index.astype(int)
        max_customer_id = customer_quantity.idxmax()
        max_customer_quantity = customer_quantity.max()
        plt.figure(figsize=(10, 6))
        print("Customer with the highest total quantity:")
        print("CustomerID:", max_customer_id)
        print("Total Quantity:", max_customer_quantity)


    def plot_sales_by_country(self, df):
        # Plot total sales by country
        total_sales_by_country = df.groupby('Country')[['Quantity', 'UnitPrice']].apply(
            lambda x: np.sum(x['Quantity'] * x['UnitPrice']))
        country_with_highest_sales = total_sales_by_country.idxmax()
        highest_sales_amount = total_sales_by_country.max()

        total_sales_by_country_sorted = total_sales_by_country.sort_values(ascending=False)
        top_10_countries = total_sales_by_country_sorted.head(10)

        ax = plt.gca()
        bars = plt.bar(top_10_countries.index, top_10_countries)

        # Annotate each bar with its value
        for bar in bars:
            height = bar.get_height()
            ax.annotate('{:.2f}'.format(height),  # Format to two decimal places
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

        # Set title and labels
        plt.title('Total Sales by Top 10 Countries')
        plt.xlabel('Country')
        plt.ylabel('Total Sales Amount')

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')

        # Tight layout to adjust spacing
        plt.tight_layout()

        # Save the figure
        plt.savefig('Output/total_sales_by_country.png')

        # Show the plot
        plt.show()

    def plot_sales_by_month(self, df):
        # Plot total sales by month
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        df['Month'] = df['InvoiceDate'].dt.to_period('M')
        total_sales_by_month = df.groupby('Month')[['Quantity', 'UnitPrice']].apply(lambda x: np.sum(x['Quantity'] * x['UnitPrice']))
        plt.figure(figsize=(10, 6))
        total_sales_by_month.plot(kind='bar', color='skyblue')
        plt.title('Total Sales by Month')
        plt.xlabel('Month')
        plt.ylabel('Total Sales Amount')
        plt.xticks(rotation=45)
        for i, sales in enumerate(total_sales_by_month):
            plt.text(i, sales, f'{int(sales):,}', ha='center', va='bottom')
        plt.subplots_adjust(bottom=0.15, left=0.1, right=0.9, top=0.9)
        plt.savefig('Output/total_sales_by_month.png')
        plt.show()


try:
    # Load dataset
    print(">>> Loading the training dataset...")
    df = pd.read_csv('Input/TrainedData.csv')

    # Initialize the classifier
    print(">>> Initializing the text classifier...")
    classifier = TextClassifier()

    # Train the classifier
    print(">>> Training the classifier...")
    print(">>> Calculating the Accuracy...")
    accuracy = classifier.train(df)
    print("\n>>> Accuracy of the classifier:", accuracy)

    # Read descriptions from the new file
    print("\n>>> Reading descriptions from the online retail dataset...")
    new_df = pd.read_csv('Input/online_retail.csv')

    # Predict categories for new descriptions
    print(">>> Processing the data...")
    predicted_df = classifier.predict(new_df)

    # Save predictions
    print(">>> Saving predictions to files...")
    classifier.save_predictions(predicted_df)

    # Plot category distribution
    print("\n\n>>> Plotting category distribution...")
    classifier.plot_category_distribution(predicted_df)

    # Calculating top customers
    print("\n\n>>> Calculating top customers...")
    classifier.Analyze_top_customers(predicted_df)

    # Plot sales by country
    print("\n\n>>> Plotting sales by country...")
    classifier.plot_sales_by_country(predicted_df)

    # Plot sales by month
    print("\n\n>>> Plotting sales by month...")
    classifier.plot_sales_by_month(predicted_df)

except FileNotFoundError:
    print("Error: File not found. Please make sure the file paths are correct.")

except Exception as e:
    print("An unexpected error occurred:", e)

print(">>> Exit")
print("\n\n>>> Thank you!")
