📉🔮 LAB 2: Predict Customer Churn in Camtel 💡📊

🧑‍🎓 Student Name: Donfack Tsopfack Yves Dylane

Please look at model3.py(adjusted model) and decision tree.py(base model)


🌟 INTRODUCTION 🌟

The goal of this project is to build a machine learning model to predict customer churn for Camtel 📡, a major telecommunications provider. 

Customer churn—the rate at which customers leave the service—is a key challenge that directly impacts revenue 💸 and business stability 🏢.

This project focuses on tackling concept drift and data shifts, ensuring that our model stays adaptive and reliable as customer behavior evolves over time. 

We'll explore the dataset, build initial models, and adapt our approach to keep the model fresh and effective! 🌱💡


Key Metrics:


Accuracy

AUC-ROC

Model performance over time ⏳


🧹 Step 1: Data Cleaning and Preprocessing 🧼

Cleaning the dataset is where the magic begins! 🧙‍♂️✨ I wrote a function called import_and_clean_data that uses pandas to:

Drop missing values 🙅‍♂️

Remove duplicate rows 🔄

Eliminate unnecessary columns like customer_id 🧾

🔄 Converting Data and Removing Outliers:

After cleaning, I transformed non-numeric data into numeric form and removed outliers based on their distributions. This makes the data ready for the next steps! 📈


📊 Correlation Graph:

We then created a correlation graph to visualize relationships between features, ensuring we're set up for accurate predictions! 🖼️


🌱 Step 2: Initial Model Training 🎯

For the initial model, I used a Decision Tree 🌳 to capture complex patterns in the data.


⚙️ Model Training:

Year 1 data was used for training 📚, while Year 2 data was for testing 🧪.

Key metrics: Accuracy, Precision, Recall, F1-Score, and AUC-ROC 📈.


⚠️ Concept Drift Detected:

As expected, model performance declined when testing on Year 2 data, indicating concept drift—the data patterns shifted from one year to the next. 😯


🧠 Step 3: Dealing with Concept Drift 🧠

To handle concept drift and data shifts, I implemented the following techniques:


🕒 3.1 Time-Weighted Learning:

We gave more recent data higher importance during training, so the model stays focused on the latest trends! 📅


🔄 3.2 Online Learning (SGD):

Using Stochastic Gradient Descent (SGD), I implemented an online learning algorithm that updates the model continuously as new data arrives, making it adaptable in real-time! 🚀


🤝 3.3 Ensemble Models:

We also combined predictions from models trained on different years using ensemble learning 🧩, capturing patterns from each time period for improved accuracy.


⚖️ Step 4: Model Evaluation and Adaptation 🔍

After testing our baseline model against the adapted models (time-weighted, online learning, ensemble):


Surprise! The baseline model showed higher precision than the adapted models in earlier time periods. 😲

This suggests the baseline model might be overfitting to older data, leading to unexpected results. (I suspect I might have done something wrong here 🧐).


🔮 Conclusion 🏁

In this project, I built a predictive model for customer churn at Camtel, addressing challenges like concept drift and data shifts. 

While the adapted models did not fully resolve these issues, the project highlights the complexity of building models that adapt to changing customer behaviors.


🌱 Future Directions:

Explore advanced drift detection methods 📊.

Consider dynamic model updates to improve adaptability over time! 🚀

This project provides a foundation for future work in building more adaptive models for churn prediction. 🌟

