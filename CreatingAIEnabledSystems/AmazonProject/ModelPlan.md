# Feedback Mechanism
- **Sentiment Analysis**: Incorporate natural language processing to analyze the sentiment of the review text. With this I am able to generate a model that returns the user sentiment based mostly on the review_title and text. The model would then send back a polarity score and place it into a category ranging from Very Negative to Very Positive. This allows the user to have a greater understanding of the review.

# Analysis Model
- **Text Models**: Utilizing tfidf I am able to convert the text into a format that is easier for another model to understand. Once it has been sent through tfidf I utilize a UMAP model to put the data into smaller categories that still give information to the main model.
- **Main Model**: A Random Forest Model is added to predict the polarity score of any text that is sent its way sending back what it believes to be positive or negative reviews.

# Integration Strategy
- **API Development**: Utilizing Postman and Docker people can send information to the model and recieve back a polarity score and category that the review falls under.
- **Other Integration**: This could be added as an automated item once a review comes in it could send directly to the model to be interpreted and then added to whatever database is needed.