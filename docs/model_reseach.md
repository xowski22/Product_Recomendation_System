Collaborative filtering
-
- past similar preferences can inform future prefs
- classify users into clusters of similar types and recommend each user according to the preference of it's cluster

Matrix Factorization 
- 
- helps to understand the hidden patterns how users interact with items
- each dimension in vector is its own feature
- it learns by multiplying vectors, thus giving us a prediction that matches the rating

Other methods of Collaborative filtering
- 
Memory based filtering
-

User based - finds similar users and assumes that they like similar things, works by calculating cosine similarity and then uses ratings to make predictions

Item based - works opposite to user based, if many users like movie1 and movie2 it assumes that these movies must be similar to each other, can be more stable then user based and can be precomputed

Model based filtering
-
Clustering - groups similar users or items together and then uses these clusters for recommendations, easy to interpret but might be disadvatagious when looking for nuanced patterns

Neural Collaborative Filtering (NCF) - replaces dot product in matrix factorization with neural network, allows to learn more complex interactions between users and items

Autoencoders - create compressed representation od user preferences or item features, forces model to learn most important patterns in data




