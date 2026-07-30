# feature engineering

## what is feature engineering

- for building better ML models
- identify most important features with mutual information
- encode high-cardinality categories using target encoding
- invent features
- segmentation features k-means clustering
- decompose dataset w/ principle component analysis

- goal of feature engineering: make data better suited to problem
  - improve predictive performance
  - reduce computational or data needs
  - improve interpretability

- e.g. apparent temperature: i.e. windchill and heat index
  - based on measureables like wind speed, humidity, temp
  - the feature engineering here is attempting to make observed data more relevant to "feelings"

- guiding principle
  - for feature to be useful: it must have learnable relationship to target
    - linear model learns linear relationships, therefore goal is to transform features to make relationship to target linear
    - e.g. if target is price and feature is length, and the price/lengths produce an exponential curve, can transform by squaring lengths (area) to make a linear relationship
    - e.g. concrete formulations
      - concrete as in the building material
      - target compressiveStrength
      - first establish "baseline score" - no changes get MAE
      - add several features - ratios of ingreedients
        - Fine/Course ratio
        - fine+course / cement ratio
        - water/cement ratio
      - with those new features, MAE decreased

## mutual information

- intro
  - new dataset -> overwhelming - thousands of features, no descriptions
  - first step: construct ranking with a feature utility metric
    - a function measuring associations between feature and target
    - pare down to smaller set of feature (take only most useful feature)
    - metric: "mutual information" - like correlation - measures relationship between two quantities
      - correlation identifies just linear relationships
      - mutual information detects any relationship
  - mutual information
    - easy to use
    - computationally efficient
    - theoretically well-founded
    - resistant to overfitting
    - can detect any relationship
- mutual info and what it measures
  - describes relationships in terms of "uncertainty"
  - measures how knowledge of one quantity reduces uncertainty of the other
  - e.g. graph - sale price vs quality of house
    - shows that prices for each quality tend to cluster around distinct ranges of prices
    - if less data availailable for a certain feature, that is weighted less
    - entropy aka uncertainty
- interpreting MI scores
  - MI score of 0 means no relationship
  - MI score of 1 means perfect relationship
  - MI score of 0.5 means that knowing one quantity reduces uncertainty of the other by half
  - MI is logarithmic
  - possible that certain feature more relevant taken together with other features, MI cannot capture this, MI is univariate
  - MI score on feature doesn't mean model works well with it - may need to transform feature to make it more useful to the model
- new data set: 1985 automobiles
  - target price
  - prepping data for MI
    - MI treats continous and discrete data differently
    - factorize() function converts categorical data to numeric
    - example ensures all features considered are int type
    - scikit learn: feature selection library
      - metrics split into two functions: mutual_info_classif() and mutual_info_regression(), for categorical and continuous targets respectively
```
from sklearn.feature_selection import mutual_info_regression

def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

mi_scores = make_mi_scores(X, y, discrete_features)
mi_scores[::3]  # show a few features with their MI scores
```
- so we see it takes in columns and target and a boolean that indicates which features are discrete, and returns a series of MI scores for each feature
- the output is a score for each column
- after you identify high MI scores, recommended to visualize those feature's distributions

### MI: exercise

- ames data source
- ranked MI scores
- investigate relation of a categorical feature: building type
- then look at building type together with feature grlivarea, and mosold

## creating features

- previous: identified features useful
- now: start developing them - will learn transformations on features
- going to use 4 datasets: US Traffic accidents, 1985 automobiles, Concrete Formulations, Customer Lifetime Value
- tips
  - understand features - read dataset docu if available
  - research problem domain and gain domain knowledge
  - study others' models
  - use data visualizations to reveal anomolies and relationships

- mathematical transformations
  - 2 or more numerical features can be transformed into new feature with math
    ```py
      autos["stroke_ratio"] = autos.stroke / autos.bore
    ```
  - more complicated combination -> more difficult for model to learn
    ```py
      autos["displacement"] = ( np.pi * ((0.5 * autos.bore) ** 2) * autos.stroke * autos.num_of_cylinders )
    ```
  - sometimes skewed data should be normalized with a logarithm
    ```py
      accidents["LogWindSpeed"] = accidents.WindSpeed.apply(np.log1p)
    ```


- counts
  - a feature that describes absence or presence of something (boolean or binary) come in sets that can be tallied (use sum())
    ```py
      roadway_features = ["Amenity", "Bump", "Crossing", "GiveWay", "Junction", "NoExit", "Railway", "Roundabout", "Station", "Stop", "TrafficCalming", "TrafficSignal"]
      accidents["RoadwayFeatureCount"] = accidents[roadway_features].sum(axis=1)
    ```
  - same technique can be used for continuous data, treating 0.0 as false and any other value as true

- build up and break down
  - complex strings can be broken down to simpler pieces, for example
    - ID numbers: '123-45-6789' -> '123', '45', '6789'
    - phone numbers: '(123) 456-7890' -> area code, number
    - street address: '123 Main St, Springfield, IL 62701' -> '123', 'Main St', 'Springfield', 'IL', '62701'
    - urls
    - product codes
    - date and time
  - can use split to help with this
    ```py
      customer[['type', 'level']] = ( customer["Policy"].str.split(' ', expand=True)) # "Corporate L3" -> "Corporate", "L3"
    ```
  - build up is just the opposite, composing multiple features into one
  - data types that require more study: dates, geolocation

- group transforms
  - aggregate info across multiple rows grouped by some category
    - e.g. "average income of person's state of residence"
      ```py
        customer["AverageIncome"] = ( customer.groupby('State')['Income'].transform('mean'))
      ```
  - available transform functions
    - mean, median, min, max, sum, count, std, var, sem, first, last, nth

  - parting tips
    - linear models learn sums and diffs naturally
    - ratios give models trouble
    - linear models do better with normalized features
      - neural nets want values close to 0
    - tree-based models can deal with approximating combination of features, but may benefit from creating combination features
    - counts helpful for tree models since they can't naturally aggregate info

## exercise: creating features

- made a mistake when doing counting feature exercise
```py
X_3["PorchTypes"] = X["WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "Threeseasonporch", "ScreenPorch"].gt(0).sum(axis=1) # incorrect
X_3["PorchTypes"] = X[["WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "Threeseasonporch", "ScreenPorch"]].gt(0).sum(axis=1) # correct
```

- made mistake when solving grouped transform with requirement "The value of a home often depends on how it compares to typical homes in its neighborhood. Create a feature MedNhbdArea that describes the median of GrLivArea grouped on Neighborhood."
```py
X_5["MedNhbdArea"] = X.groupby("Neighborhood")['GrLivArea'].transform('mean') # incorrect
X_5["MedNhbdArea"] = X.groupby("Neighborhood")['GrLivArea'].transform('median') # correct
```

## cluster with k-means

- intro
  - will talk about unsupervised learning algorithms
    - a "feature discovery" technique
    - doesn't use target
    - learn property of data
  - clustering
    - group data points into clusters based on similarity
    - k-means clustering is a popular algorithm for this
    - examples
      - groups of customers representing market segment
      - geo areas with similar weather patterns
- cluster labels as feature
  - applied to single feature - traditional "binning"
  - applied to multiple features - "multi-dimensional binning"
  - studying some illustrations and tables
    - it appears we add a categorical feature "cluster" whose values are the cluster number, assigned to each data point
    - also can use one-hot encoding
    - clustering can break a curve into chunhks of linear relationships, which is useful for linear models
- k-means clustering
  - there are many clustering algorithms
  - differ in how they measure similarity and proximity
  - k-means clustering
    - measures similarity by euclidean distance
    - puts "centroids" or points around the feature space
    - the centroid they are closest to is the cluster they belong to
  - impl: we focus on 3 params (scikit-learn)
    - n_clusters: number of clusters to form
    - max_iter: maximum number of iterations of the k-means algorithm for a single run
    - n_init: number of times the algorithm will run with different centroid seeds, and the best output is chosen
    - 2 step loop (which max_iter controls)
      1.  assign points to nearest cluster centroid
      2.  move centroids to minimize distance to points
    - as the algorihtm iterates, points shift between clusters
    - normally we only are concered with n_clusters
- example - california housing
  - latitude and longitude are good candidates for KMC,
  - cluster them with medInc (median income) to create clusters
  - so 3 features used, but as many features as you want can be used
  ```py
    X = df.loc[:, ["MedInc", "Latitude", "Longitude"]]
    # Create cluster feature
    kmeans = KMeans(n_clusters=6)
    X["Cluster"] = kmeans.fit_predict(X)
    X["Cluster"] = X["Cluster"].astype("category")

    X.head()
  ```

## exercise - scaling features

- frst step: setup
  - defines score_dataset
  - reads csv file

- scaling features
  - mentions that KMC is sensitive to scale of features
  - asks us to examine some features and termine if they should be rescaled
    - latitude and longitude - no - because don't want to distort distances
    - lot area and living area - either way - living area more valuable per area
    - number of doors and horsepower - should be scaled - no comparable units - num of doors would have negligible effect on weighting compared to horsepower

- create feature of cluster labels
  - tells us to create k-means clustering with some params with 10 clusters and 10 iterations
  - made a mistake - used max_iter=10 instead of n_init=10

- cluster-distance features
  - introduces new concept - distance to cluster centroid
    - use fit_transform to utlize this