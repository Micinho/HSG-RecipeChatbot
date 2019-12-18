# Import necessary modules:
import requests
import nltk
import numpy
import tflearn
import tensorflow
import random
import json
import pickle
import time
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

# Define api key for spoonacular database
api_key = "PLEASE_INSERT_YOUR_API_KEY_HERE"

# Create empty lists to store the current recipe id, the recipe ids of all recipes retrieved as well as their titles
currentRecipeID = []
recipeID = []
recipeTitle = []

# Load data: open intents file to access input data for chatbot
with open("spoonacular_intents.json") as file:
    data = json.load(file)

# If input data (json file) has already been processed, open the file with the processed data
# If input data has not been processed before: create the necessary lists for the chatbot to train itself
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)

except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    # Loop through intents and extract the relevant data:
    # Turn each pattern into a list of words using nltk.word_tokenizer,
    # Then add each pattern into docs_x list and its associated tag into the docs_y list
    # Also, add all individual intents to the labels list
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    # Get data ready to feed into our model with the help of stemming, and sort words and labels
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    labels = sorted(labels)

    # Create "bag of words" to turn list of strings into numerical input for machine learning algorithm:
    # Create output lists which are the length of the amount of labels/tags we have in our dataset
    # Each position in the list will represent one distinct label/tag,
    # A 1 in any of those positions will show which label/tag is represented
    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w.lower()) for w in doc]
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1
        training.append(bag)
        output.append(output_row)

    # Convert training and output lists into numpy arrays
    training = numpy.array(training)
    output = numpy.array(output)

    # Save processed data, so that it does not have to be processed again later
    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

# Setup of our model
tensorflow.reset_default_graph()

# Define network, two hidden layers with eight neurons
# Input data with length of training data
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

# Define type of network (DNN) and take in designed network (from above)
model = tflearn.DNN(net)

# If model has already been fitted, load the model
# Otherwise: fit the model --> pass all of training data
try:
    model.load("model.tflearn")

except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

# Make predictions
# First step in classifying any sentences is to turn a sentence input of the user into a bag of words
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)

# Second step: code that will ask the user for a sentence and then spit out a response, in case user did not quit
def chat():
    time.sleep(0.5)
    print("Recipe bot: Ask me questions about the chosen recipe.")
    time.sleep(0.5)
    print("Recipe bot: You can ask me about the ingredients, the cooking steps, the recipe's nutrition or simply ask to go back or make a new search.")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        # If confidence level is higher 85%, open up the json file, find specific tag and spit out response
        if results[results_index] > 0.85:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
            print("Recipe bot:", random.choice(responses))

            # Link chatbot (intent of user) to respective function in program
            if responses == ["These are the ingredients:"]:
                getRecipeIngredients(str(currentRecipeID[0]))
            elif responses == ["These are the instructions:"]:
                getRecipeInstructions(str(currentRecipeID[0]))
            elif responses == ["Welcome (back) to the overview:"]:
                chooseRecipe()
            elif responses == ["Here, you can start again with new ingredients:"]:
                currentRecipeID.clear()
                recipeID.clear()
                recipeTitle.clear()
                getRecipeByIngredients()
            elif responses == ["You are welcome!"]:
                chat()
            elif responses == ["See the recipe's nutrition information:"]:
                getRecipeNutrition(str(currentRecipeID[0]))
        else:
            print("Recipe bot: I am not sure what you want to do. Can you rephrase your question?")
            chat()

# Define main function: Retrieving recipes based on ingredients (input)
def getRecipeByIngredients():
    print("Recipe bot: Please enter between 1 and 10 ingredients that you want to use for a recipe. Enter them in singular form separated by a space (for example: milk egg chicken).")
    ingredients = input("You: ")
    # Checking if user entered between 1 and 10 ingredients, otherwise start again
    if len(ingredients.split()) <= 10 and len(ingredients.split()) >= 1:
        ingredients = ",".join(ingredients.split(" "))
        print("Recipe bot: How many recipes would you like to see? Choose an integer between 1 and 10")
        try:
            numberOfRecipes = int(input("You: "))
        except ValueError:
            print("Recipe bot: You have not chosen between 1 and 10 recipes. Starting again...")
            getRecipeByIngredients()

        # Defining the payload sent to the API
        if 1 <= numberOfRecipes <= 10:

            payload = {
                'ingredients': ingredients,
                'number': numberOfRecipes,
                'ranking': 1,
                'fillIngredients': False,
                'limitLicense': False
            }

            # The API endpoint
            endpoint = "https://api.spoonacular.com/recipes/findByIngredients?apiKey="+api_key

            # GET request to the API. Save titles and IDs of recipes in lists and print recipe titles
            r = requests.get(endpoint, params=payload)
            results = r.json()
            n = 0
            for i in results:
                recipeTitle.append(results[n]["title"])
                recipeID.append(results[n]["id"])
                n += 1

            # If recipes are available, call the chooseRecipe() function
            # If recipes are not available, print a message
            if len(recipeTitle) >= numberOfRecipes:
                chooseRecipe()
            elif len(recipeTitle) == 0:
                print("Recipe bot: There are no recipes available for the entered ingredients. Please start again...")
                getRecipeByIngredients()
            else:
                print("Recipe bot: There are only",len(recipeTitle),"recipes available.")
        else:
            print("Recipe bot: You have not chosen between 1 and 10 recipes. Starting again...")
            getRecipeByIngredients()
    else:
        getRecipeByIngredients()

# Create function to choose one of the retrieved recipes
def chooseRecipe():
    n = 0

    # Print recipe titles
    print("")
    for i in recipeTitle:
        print(n + 1, recipeTitle[n])
        n += 1
    print("")
    print("Recipe bot: Choose a recipe by typing in the recipe's number.")

    # Check input (integer) and show respective recipe after selection, otherwise restart function
    try:
        choiceOfRecipe = int(input("You: "))
    except ValueError:
        chooseRecipe()
    if 1 <= choiceOfRecipe <= len(recipeTitle):
        currentRecipeID.clear()
        currentRecipeID.append(recipeID[choiceOfRecipe-1])
        print("Recipe bot: You chose",recipeTitle[choiceOfRecipe-1],"- good choice!")
        chat()
    else:
        chooseRecipe()

# Create function to retrieve the instructions for a specific recipe
def getRecipeInstructions(id):

    # Defining the payload sent to the API
    payload = {
        'id': id,
        'stepBreakdown': False
    }

    # The API endpoint
    endpoint = "https://api.spoonacular.com/recipes/" + str(id) + "/analyzedInstructions?apiKey=" + api_key

    # GET request to the API. Save instruction steps (step_name) and print steps and their respective sub steps
    r = requests.get(endpoint, params=payload)
    results = r.json()

    n = 0

    # State the name of the steps
    for step in results:
        print("\nStep", n + 1, ":", results[n]["name"])

        # List all sub-steps
        for sub_step in results[n]['steps']:
            sub_steps = sub_step['step']
            sentenceSplit = sub_steps.split(".")

            for s in sentenceSplit:
                if not s:
                    continue
                if s[0] == " ":
                    b = s.replace(" ", "", 1)
                    if b[0] == " ":
                        c = b.replace(" ", "", 1)
                        print(" – ", c + ".")
                    else:
                        print(" – ", b + ".")
                else:
                    print(" – ", s + ".")
        n += 1
    print("")
    chat()

# Create function to retrieve the ingredients of a specific recipe
def getRecipeIngredients(id):

    # Payload sent to the API
    payload = {
        'id': id
    }

    # The API endpoint
    endpoint = "https://api.spoonacular.com/recipes/" + str(id) + "/ingredientWidget.json?apiKey=" + api_key

    # Get request to the API. Save name, weight and unit of a specific ingredient and print all ingredients
    r = requests.get(endpoint, params=payload)
    results = r.json()
    ing_name = []
    ing_wei = []
    ing_unit = []
    n = 0
    for ingredient in results["ingredients"]:
        ing_name.append(results["ingredients"][n]["name"])
        ing_wei.append(results["ingredients"][n]["amount"]["metric"]["value"])
        ing_unit.append(results["ingredients"][n]["amount"]["metric"]["unit"])
        n += 1
    n = 0
    print("")
    for ingredient in range(len(ing_name)):
        print(ing_wei[n], ing_unit[n], ing_name[n])
        n += 1
    print("")
    chat()

# Create function to retrieve the nutrition details of a specific recipe
def getRecipeNutrition(id):

    # Payload sent to the API
    payload = {
        'id': id
    }

    # The API endpoint
    endpoint = "https://api.spoonacular.com/recipes/" + str(id) + "/nutritionWidget.json?apiKey=" + api_key

    # Get request to the API. Print nutrition results
    r = requests.get(endpoint, params=payload)
    results = r.json()
    print("\ncalories:",results["calories"],
          "\ncarbs:",results["carbs"],
          "\nfat:",results["fat"],
          "\nprotein:",results["protein"],"\n"
          )
    chat()

# Initiate chatbot by calling the getRecipeByIngredients function
print("Recipe bot: Hi! I am your recipe bot and will help you to find your perfect recipe!")
getRecipeByIngredients()
