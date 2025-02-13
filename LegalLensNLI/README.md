#### How to Run the Project

To begin, ensure that you have installed all the required packages. It is recommended to use **python 3.10.10**.

##### Step 1: Install Required Packages

Run the following command to install all necessary dependencies:

```
pip install -r requirements.txt
```

##### Step 2: Model Training (Optional)

If you wish to retrain the model, please refer to the 'training.ipynb' notebook. Simply run the provided scripts.

We have included all the necessary data files within the source folder, so you do not need to modify the code. However, if you plan to use your own data, please ensure that the format of your new files matches the expected structure in the code.

##### Step 3: Load Checkpoint and Infer Test Set

We have uploaded the final model checkpoint to Hugging Face (the endpoint is shown in the model loading code). The final prediction file, `predictions_NLILens.csv`, was generated by running the code in the `legallens-infer-final.ipynb` notebook. We also leave the access token in the notebook: `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`
