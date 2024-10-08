# Car Price Prediction System

This project implements a machine learning model to predict the selling price of used cars based on various features. The dataset is cleaned and processed to remove unnecessary features, and a Random Forest Regressor is used for predictions. The model is trained on a refined dataset after removing irrelevant columns and filling missing values.

## Project Overview
The dataset used consists of various car specifications that influence the selling price of used cars. Key steps such as data cleaning, feature selection, and model training are performed to build a predictive model.

# Key Features:

**Model Used:** Random Forest Regressor

**Metrics Evaluated:**
R² Score, MAE, MSE and RMSE

## Dataset

• Source: https://github.com/Alpha107/Car-Price-Prediction-System/blob/main/CarsDataset.csv

• Size: Dataset with multiple rows and various features such as car specifications

• Target Variable: Selling Price (Numeric)

## Features Used for Model Building:
After feature selection, the following features were retained and used for model training:

• Car Model: Specific model of the car.

• Car Brand: Brand of the car.

• Year of Manufacture: Year the car was manufactured.

• Mileage (Highway/City): Car mileage on highways or in the city.

• Horsepower: Power of the car engine.

• Number of Seats: Number of seats available in the car.

• Transmission Type: Type of transmission system (Automatic/Manual).

• ABS (Anti-lock Braking System): Indicates if the car has ABS.

• Fuel Type: Type of fuel used by the car (e.g., Petrol, Diesel).

• Central Locking: Availability of a central locking system.

• Engine Malfunction Light: Indicator for engine malfunction.

• Child Safety Locks: Child lock availability.

• USB Compatibility: Indicates if the car has USB compatibility.

• EBD (Electronic Brake-force Distribution): Availability of electronic brake distribution.

• EBA (Electronic Brake Assist): Presence of electronic brake assist.

• Number of Doors: Total number of doors in the car.

• And many more.

These features, after encoding, were essential in determining the car's selling price.

## Libraries and Tools Used
• Python (3.9.12)

• Pandas

• NumPy

• Matplotlib

• Seaborn

• Scikit-learn

## Steps Involved
**Data Preprocessing:**

• Load the dataset and explore the columns.

• Remove irrelevant features based on analysis, and fill missing values with the mean for numeric features or the mode for categorical features.

• Apply label encoding to categorical columns and one-hot encoding for certain columns.

**Train-Test Split:**

•Split the data into 80% training and 20% testing sets.

**Model Training:**

•Train a Random Forest Regressor on the training data.

•Use the trained model to predict the selling price on the test set.

**Evaluation:**

• Metrics such as R² Score, MAE, MSE and RMSE are used to evaluate the model's performance.

## Results

• Model	R² Score: 89.96%
This indicates that your model explains 89.97% of the variance in the target variable (selling price). This is a very good score, showing that your model performs well at predicting car prices.

• MAE:  1.5519
On average, your predictions are off by about 1.5519 units. Depending on the currency or scale of the selling price, this could indicate a fairly small or significant error. For example, if prices are in thousands of dollars, an error of 1.5516 might be quite low.

• MSE: 10.7460
This value is the average squared error. It is harder to interpret directly, but smaller values indicate better performance.

• RMSE: 3.2781
This metric is easier to interpret because it's in the same units as the target variable. A lower RMSE means that, on average, your model's predictions are 3.2781 units off from the actual values.


## Prediction

The system predicts the selling price of used cars based on various input features. For example:

<pre>
<code class = "language-python">
import numpy as np

car_features = np.array([1, 1, 0, 0, 1, 1, 0])  # Sample features for a car

# Reshape the input features to match the model input shape
predicted_price = regressor_rf.predict(car_features.reshape(1, -1))

print(f"Predicted Selling Price: {predicted_price}")
</code>  
</pre>

  
## Visualizations

• Correlation Heatmap: Displaying relationships between numerical features.

![Correlation](https://github.com/user-attachments/assets/673f8f6d-7b3a-430f-a336-91838fcffd16)


• Missing Values Heatmap: Visualized for identifying missing values.

Before:

![MissingValues](https://github.com/user-attachments/assets/dd89db20-851a-4945-a641-94bb8fe7e743)

After:

![NoMissing](https://github.com/user-attachments/assets/133d8189-75b2-48fb-a066-d1165b91fd85)


• VIF values of each features: 

**About:** Variance Inflation Factor (VIF) is a measure used to quantify how much the variance of an estimated regression coefficient increases when your predictors are correlated. In simpler terms, VIF helps you understand if your independent variables are highly correlated with each other, which can lead to multicollinearity issues in regression analysis.

- VIF = 1: No correlation between the predictor and other variables (not problematic).

- 1 < VIF < 5: Moderate correlation (generally acceptable).

- VIF >= 5: Indicates high correlation (potentially problematic).

- VIF >= 10: Indicates serious multicollinearity issues (consider removing or combining variables).

For our Dataset characteristics:

1                                       Model --> 1.266742

2                                     Variant --> 1.335636

3                           Ex-Showroom_Price --> 1.319288

4                                Displacement --> 4.762672

5                                   Cylinders --> 3.885365

6                         Valves_Per_Cylinder --> 1.136321

7                                  Drivetrain --> 1.478662

8                      Cylinder_Configuration --> 3.402267

9                               Emission_Norm --> 1.505417

10                            Engine_Location --> 1.372148

11                                Fuel_System --> 1.301426

12                         Fuel_Tank_Capacity --> 2.642576

13                                  Fuel_Type --> 1.701360

14                                     Height --> 3.849863

15                                      Width --> 5.217953

16                                  Body_Type --> 1.938959

17                                      Doors --> 2.711373

18                     ARAI_Certified_Mileage --> 1.698861

19                                      Gears --> 2.796457

20                           Ground_Clearance --> 1.901466

21                               Front_Brakes --> 1.341007

22                                Rear_Brakes --> 3.969603

23                           Front_Suspension --> 1.646302

24                            Rear_Suspension --> 1.595503

25                             Power_Steering --> 1.283326

26                              Power_Windows --> 1.409212

27                                      Power --> 1.809936

28                                     Torque --> 5.180067

29                                   Odometer --> 1.982969

30                                Speedometer --> 1.903107

31                                 Tachometer --> 2.129241

32                                  Tripmeter --> 1.234073

33                           Seating_Capacity --> 2.827046

34                                       Type --> 1.994861

35                           12v_Power_Outlet --> 1.272344

36                                Audiosystem --> 1.371051

37                             Basic_Warranty --> 1.508076

38                            Boot-lid_Opener --> 1.301310

39                                 Boot_Space --> 1.848917

40                            Central_Locking --> 2.036339

41                         Child_Safety_Locks --> 1.441016

42                                      Clock --> 1.521939

43                               Door_Pockets --> 1.493162

44                   Engine_Malfunction_Light --> 1.507990

45                            Fuel-lid_Opener --> 1.220966

46                                 Fuel_Gauge --> 1.648566

47                      Multifunction_Display --> 1.322629

48             ABS_(Anti-lock_Braking_System) --> 3.450323

49  EBD_(Electronic_Brake-force_Distribution) --> 3.394709

50                          Number_of_Airbags --> 2.985565

51                          USB_Compatibility --> 2.251737

52              EBA_(Electronic_Brake_Assist) --> 2.529375

53                     Seat_Height_Adjustment --> 2.132416


## Future Improvements

• Add more advanced machine learning models like Gradient Boosting or Neural Networks to improve prediction accuracy.

• Fine-tune hyperparameters of the Random Forest model for better performance.

• Incorporate additional relevant features into the dataset.
