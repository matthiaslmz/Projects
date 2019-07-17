[Website Link](https://statisticcanada.github.io/)

Training Data: 
* `Combined_Links.csv` contains the 417 record data set that we have manually categorized. This data set would act as our training data for all the text classifiers used in the notebooks. The columns contain: Title, Description, Link, Category, Subcategory, Source 

New data (To be classified): 
* `New Data.csv` Data set that has been scrapped and not labeled, we be categorized using the ensemble classifiers.

Classified New Data: 
* contains the categorized data set using the New data mentioned above. One filed named `combined.csv` uses the method of merging the all the predicted class of the base classifiers and dropped its duplicates. This method is not recommened and just used for or own research purposes. 

* The `Bestcombined.csv` is the best ensemble classifier classified data set. Contains 1376 records in total, 1033 in supply, 152 in consumption, 88 in Health Outcome, 44 in Distribution, and 59 in Utilization. The data set was examined for accuracy, we've realized that the 1033 supply category had some minor errors. It is categorizing data products that essentially is supply such as GDP, imports and exports, but not relevant to Food and Nutrition. This may be improved in the future. 




