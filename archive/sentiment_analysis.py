import sys
from textblob import TextBlob

def analyze_sentiment(text):
    """
    Analyzes the sentiment of a given text and returns 'positive', 'neutral', or 'negative'.
    """
    blob = TextBlob(text)
    
    # polarity is between -1.0 and 1.0
    polarity = blob.sentiment.polarity
    
    if polarity > 0.1:
        return "positive"
    elif polarity < -0.1:
        return "negative"
    else:
        return "neutral"

def main():
    # If a file is passed as an argument, use it; otherwise, read 'large_text_file.txt'
    filename = sys.argv[1] if len(sys.argv) > 1 else 'large_text_file.txt'
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read()
            
        print(f"Analyzing sentiment for {filename}...")
        sentiment = analyze_sentiment(text)
        print(f"The sentiment of the text file is: {sentiment}")
        
    except FileNotFoundError:
        print(f"Error: The file {filename} was not found.")
        print("Please create the file or pass a valid filename as an argument.")
        print("Example: python sentiment_analysis.py my_text_file.txt")
        
        # Create a sample text file to demonstrate
        print("\nCreating a sample 'large_text_file.txt' for demonstration...")
        sample_text = "I absolutely love this library! It makes sentiment analysis so easy and wonderful. However, sometimes it is a bit slow and terrible, but overall I am very happy."
        with open('large_text_file.txt', 'w', encoding='utf-8') as f:
            f.write(sample_text)
        print("Sample file created. Run the script again to analyze its sentiment.")

if __name__ == "__main__":
    main()
