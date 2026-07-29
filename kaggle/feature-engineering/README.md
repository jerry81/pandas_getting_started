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

-