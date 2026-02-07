<h1 style="text-align:center;">Neural Network Models for Electrocardiogram Classification</h1>
<h2 style="text-align:center;">Final Project Report</h2>
<h3 style="text-align:center;"> Austin Bostwick, Nicholas Emer, Jack Lattanzi, and Nate Peebles</h3>

# Abstract (Motivation/Relevance)
Atrial and ventricular fibrillations are well-known risk factors for stroke due to the potential formation of blood clots in the heart that can travel to the brain. Detecting heart arrhythmias early before they devolve into fibrillation through an ECG can lead to the initiation of anticoagulant therapy, which significantly reduces the risk of stroke. Effective ventricular and atrial arrhythmia management can significantly improve a patient's quality of life. It can reduce symptoms such as palpitations, shortness of breath, and fatigue, allowing individuals to maintain a more active and fulfilling lifestyle.
Detection of atrial and ventricular arrhythmias involves analyzing an ECG signal. This process typically requires a highly trained cardiologist who interprets the ECG readings to identify abnormalities. Upon detecting an irregularity, the cardiologist assesses its origin and formulates a treatment strategy tailored to the abnormality's severity and the patient's overall health condition. This can be a time-consuming process, where developing a classification neural network could help streamline the diagnosis process. 
Implementing a neural network for analyzing ECG data could revolutionize real-time diagnostic support, especially in emergencies where quick decision-making is essential. Such automated diagnosis could be invaluable in regions with limited access to cardiologists, enhancing healthcare by delivering quality diagnostics without necessitating an on-site specialist. A remote specialist could receive the preliminary diagnosis made by the neural network and confirm its diagnosis by investigating the ECG segment of interest. The time of diagnosis would be decreased because the classification neural network could send packaged sections of an ECG waveform to a remote cardiologist to make a final diagnosis. Moreover, this could contribute to cost-efficiency within the healthcare system and enhance early detection rates, thereby preventing more significant health issues. 

# Scientific Context
The Pan-Tompkins algorithm is the most widely used QRS complex detector in industry. There are other algorithms based on Pan-Tompkins, such as the Hamilton-Median. There are other algorithms in general, such as RS Slope, Sixth power, “jqrs”, and many more. While a more complicated algorithm may produce better results, the Pan-Tompkins algorithm can be designed quickly and with the same results as those other algorithms. 
At the same time, a fully connected feed-forward neural network was used in the system's design. This is a one-directional information-flowing network that only flows forward with no loops or cycles. This is common for classification networks, which can classify photos or data. This is different from Recurrent Neural Networks, which have loops or reuse data from themselves. The feed-forward network is more suited for predicting a single output, which is what is being done with this project. 

# Methods
<figure>
        <img src="https://github.com/AustinBostwick/ECG-Artificial-Intelligence-Diagnostic/blob/main/BIO431_Pictures/FlowChart.png" alt="FlowChart">
</figure>

# Pan-Tompkins
## Purpose of Algorithms:
Three Pan-Tompkins algorithms were developed to extract various waves' indices, amplitude, and time from an ECG signal. The information gathered by these algorithms was vital in producing the feature engineering table used to train and test the neural network. These algorithms were also the first step in signal processing, taking in the raw data of over 20,000 annotated instances from the MIT-BIH Superventricular Arrhythmia  Database (Goldberger, 2000). Multiple algorithms were made so each could be tuned to find specific arrhythmias or waves correctly, as it was not feasible to create a single algorithm that could identify the waves correctly for every arrhythmia. Multiple algorithms were also needed because it was the best avenue to extract helpful information for the feature table. 
Each algorithm follows the standard Pan-Tompkins procedure developed by J. Pan and W. J. Tompkins in 1985. The general procedure is given below. The appendix provides the code and some descriptions. Descriptions of each algorithm are also given below. 

### Main Steps in the Pan–Tompkins Algorithm

1. **Preprocessing (filter the raw signal data)**  
   Apply a bandpass Butterworth filter to isolate the frequency range typical of the heart’s QRS complex and reduce noise.

2. **Take the derivative and square the signal**  
   Compute the derivative of the filtered signal to emphasize rapid changes (slopes) in the ECG. Square the result to accentuate peaks and further suppress smaller fluctuations.

3. **Integrate the signal using a moving window**  
   Use a moving window integrator to smooth the squared signal. This step summarizes values within a window to produce a clearer signal by averaging variations.

4. **Threshold and window the signal**  
   Apply a threshold to identify significant peaks. Only values above this threshold are considered potential R-wave (or T-wave) peaks.

5. **Save peak locations and amplitudes**  
   Detect rising edges in the thresholded signal corresponding to R-wave locations, and return the peak amplitudes and their positions in the signal.

# Normal R-Wave Detection Algorithm:

The first Pan-Tompkins Algorithm was created to detect and mark the locations of the R-waves in the given signal window. The standard Pan-Tompkins procedure, given by J. Pan and W. J. Tompkins was used to provide this information. This algorithm targeted R-waves in normal and Supraventricular Premature (SP) beat signals because they display relatively normal QRS complexes. The dataset was mostly normal R-waves, so this algorithm had to be highly specific to clarify the arrhythmias. 
The appendix shows that the code starts by loading the data values from the window of interest. Then, steps 1-3 were performed to process the signal to be ready for thresholding. Next, the threshold was set to grab peaks above 0.2 in the integrated signal. This gave the exact locations of the R wave peaks. Finally, the R-wave peaks and locations were saved and returned to the feature engineering function to allow calculation. See the images below for the steps of this process on a normal signal. 

<figure>
        <img src="https://github.com/AustinBostwick/ECG-Artificial-Intelligence-Diagnostic/blob/main/BIO431_Pictures/Figure1.png" alt="Figure 1 ">
</figure>
<figure>
        <img src="https://github.com/AustinBostwick/ECG-Artificial-Intelligence-Diagnostic/blob/main/BIO431_Pictures/Figure2.png" alt="Figure 2">
</figure>

When implemented in the feature extraction, the windowing was adjusted to target the middle of the peak (see the images in the results section). The specified database had a great signal collection, and the raw data did not require extensive filtering for powerline interference or baseline wandering.

## T-wave detection in Normal T-wave signals:
The next algorithm’s main goal was to extract the T-waves' location, time, and amplitude for standard T-wave signals. The overall method for this code was to run the first algorithm to find the R-waves and then adjust the windowing to cover the entire R-wave. Once this was done, the window was masked, and the R-waves were deleted. Once these were gone, another Pan-Tompkins was run on the data without R-waves to make the T-wave the most prominent. In this case, the thresholding was related to the averaged integrated signal when it was scaled. Once this was done, this function returned the locations of the T-waves. The T-wave information for the feature table was necessary to help the neural networks find arrhythmias. 

## Abnormal T-wave detection:
The final algorithm was focused on identifying and flagging abnormal T-waves. This technique is useful for targeting any ventricular issues. Premature Ventricular Contraction (PVC) was targeted with this algorithm due to the T-wave being larger in amplitude and broader from the premature firing of a ventricle. This signal can be grabbed using the same method as above, but this time, the thresholding was more specific in finding the sizable T-wave peak created by PVC. If the signal window was found to have an abnormal T-wave, then the value was counted. True or false flags were constructed in the feature engineering table to help detect abnormal T-waves.
Before adding these functions to the feature engineering function and neural network training, they were tested and calibrated on 15 patients and over 1000 events to ensure the peak detection was highly accurate. Due to the functions' overlapping procedures, it was ensured that the small changes between them achieved the desired outcome. 

## Connection to Feature Engineering Table:
When the feature table identified a window of interest in the ECG signal using an annotation, this data window was passed into all three of the Pan Tompkins algorithms. Two functions return the locations and amplitudes of the R and T wave peaks in that window, and the third flags whether or not an abnormal T wave was found. All of the information gathered from the three algorithms was fed back to the feature engineering function to calculate the features implemented in training and testing. More details on the construction of this table for each window and patient are given in the following section. 

## Feature Engineering Table
One of the main objectives of this project was to create a feature table with several parameters derived from the Pan-Tompkins algorithms. As discussed above, the Pan-Tompkins detect Normal and abnormal R-waves and T-waves. By identifying each R-wave and T-wave, we can gather data from each. 
For the R-wave, we collected data regarding the mean R-wave amplitude, the standard deviation of the R-wave amplitude, the average R-to-R-wave interval, and the variance of the R-to-R-wave interval. These data points can tell us a lot of information that could lead to either a diagnosis or a point of concern for the doctor. We can also look at ventricular tachycardia, where the ECG signal has a wide QRS complex. A circulating wave in the heart causes this. If our R wave interval is very high, then there is the concern of the QRS being very wide. 
For the T-wave, we collected data on the average T-to-T-wave interval, the mean T-wave amplitude, the variance of the T-to-T-wave interval, and the standard deviation of the T-to-T-wave interval. Premature Ventricular Contraction was targeted because the T-wave was larger in amplitude and wider due to the premature firing of a ventricle. Therefore, we can spot this abnormality using the standard deviation between the T-to-T-wave and the T-wave amplitude.
The event types in the last column of the table are for the neural network validation. The event types in the feature engineering table are from the MIT-BIH Supraventricular Arrhythmia  Database (Goldberger, 2000). The event type is an annotation from the MIT database correlated with the ECG signal data. The annotations are N for normal, V for premature ventricular contractions, and S for supraventricular premature or ectopic beat (atrial or nodal). The annotation has a timestamp for each event that occurs. We use the beginning and ending index to create a window in which the event has occurred, which has been calculated from the annotations. This allows the neural network to look within that index to learn what event type is occurring. After the signals and annotations are inputted we remove the ~, | , and F labels from the data. The ~ and | are annotations of an unknown wave. And F denotes fusion, but since there was only one case of fusion, we removed it. We removed these because these events were found to be skewing our data and were of such low occurrence they did not make a large impact. Also, because we are training the neural network based on known components, training on unknowns isn’t applicable for this network. 
To maximize our training data, we used multiple patients' ECG data. We only used patient data containing annotations to test our neural network's accuracy. This also diversified our data, so there were different total amplitude values, average standard deviations, and abnormalities. The table below is an example of what the table looks like. The results of this table will be discussed more later in the paper. 

| Patient ID | Beginning Index | Ending Index | Mean_R_Amplitude | STD_R_Amplitude | Average_RR | Variance_RR | Mean_RR_Time | Mean_T_Wave_Amplitude | STD_T_Wave_Amplitude | Abnormal T Wave Height | Event_Type |
|------------|------------------|--------------|------------------|------------------|------------|-------------|--------------|-----------------------|----------------------|------------------------|------------|
| 800m       | 6009             | 6268         | 0.915            | 0.34828          | 0.93359    | 0.00027     | 1.49218      | 0.07                  | 0                    | 0.27478                | FALSE      |
| 800m       | 6128             | 6389         | 1.07333          | 0.04195          | 0.94140    | 3.05e-5     | 1.91406      | -0.095                | 0                    | 0.02828                | FALSE      |


# Compact Neural Network Training Methods
The first neural network design we trained was a compact neural network created using the Matlab deep learning toolbox. The main function of setting up the neural network was fitcnet(). This created a streamlined, compact network. Most of the options for this neural network were set to the defaults.  The way we set it up was to have it create a classification model with predictors X and the class labels Y. The default functions were done first to see how the network performed with the feature table data. 
## Architecture: 
Our default network for this portion was a fully connected, feed-forward neural network for classification. The size of each fully connected layer was 10, using relu activation functions and with a validation frequency of 1. 
This simple network architecture helped us get a feel for how the initial outcomes were looking. 
## Code explanation:
To start its training, the feature table was passed in. The code then removes rows from the engineer that contain certain unwanted characters ('~,' '|,' 'F') in the last column. This step cleans the data by excluding potentially problematic entries. Then, the data was randomized to prevent any order bias. The model was then trained using the critical measurements from the features table (R-wave and T-wave features). These features are inputs to the model, and it learns to predict the 'Event_Type,' which is the outcome variable and the heart condition of interest. Training was done using fitcnet(). After training, the model's accuracy is assessed first on the training set and then on the test set. This evaluation of the predict() function checks how well the model performs and helps understand its effectiveness in predicting new data.  The accuracy was calculated from the predictions, and confusion charts were used to visualize the training and testing results. See the appendix for the code for this portion.

<figure>
        <img src="https://github.com/AustinBostwick/ECG-Artificial-Intelligence-Diagnostic/blob/main/BIO431_Pictures/Figure3.png" alt="Figure 2">
</figure>

As seen in the figure above, there were three situations it looked to classify. It looked at 16,872 events for training and testing after the unnecessary events were excluded. The activation functions were relu, with ten neurons per layer (LayerSizes). This was a relatively small network in terms of hidden layer depths. The training validation patience was set to 6, meaning it would stop once the network was trained for six epochs of minimal improvement. This helped avoid overtraining. The results were based on when the validation patience was reached. 
  
# Neural Network Training and Testing
The team moved on to a larger and more customizable neural network architecture to have more control and achieve more accurate classifications. The larger number of neurons compared to the compact network were in place to make the training process take fewer epochs and find better accuracy. 

<figure>
        <img src="https://github.com/AustinBostwick/ECG-Artificial-Intelligence-Diagnostic/blob/main/BIO431_Pictures/Figure4.png" alt="Figure 4">
</figure>
## Code explanation: 
Figure 4 above shows the architecture for the Neural Network model used in this program with the full code seen in Appendix F. The first is an input layer taking the 9 features or “Predictor Names” from the table. The fully connected layer has all nine features connected to each of the 32 layers. The batch normalization layers are there for normalization and scaling to speed up and optimize the network's training. The Tanh, hyperbolic tangent layer, is the activation function similar to that of a sigmoid, except it maps input values from -1 to 1 instead of 0 to 1, and it is symmetric about the origin, while the sigmoid is not symmetric. The softmax function, used in the softmax layer, is an activation function that transforms the raw output scores of a neural network into a probability distribution over multiple classes, which is specifically important in our three option classifier.  


# Results
## Expected Outcomes
The overall expected outcome was that all the signal processing systems and the neural network combine to produce an effective classification for normal, SV, and PVC beats. We aimed to achieve at least 70% accuracy in all three categories while avoiding over-training. This entire neural network was not expected to be ready for clinical applications by many different hospitals and patients. Still, it was expected to show promise that it could lead to that with much more tuning and engineering. 
All signal processing parts were expected to come together, as shown in the flow chart. The three peak detection algorithms were expected to be sent a data window to analyze, finding the R and T wave locations and values. These values were then expected to be processed correctly to create the feature engineering table, which would then be used to train and test the neural network. The neural network was expected to reach accurate results without overtraining. The results and analysis for each part of the project are given in the following sections. 
## Pan Tompkins
The Pan Tompkins Algorithms were formatted as functions to return information to the feature engineering function and allow it to construct the feature table. Through testing the three algorithms, there was code to plot and check the results of the functions in marking different waves. These results are given in the following figures. 

<figure>
        <img src="https://github.com/AustinBostwick/ECG-Artificial-Intelligence-Diagnostic/blob/main/BIO431_Pictures/Figure5.png" alt="Figure 5">
</figure>
The above shows the accurate detection of the largest part of the peak of the R and T waves for a normal signal. The windowing and thresholding were adjusted to detect only the middle of the targeted wave. These results build confidence in the normal R and T wave detection algorithms and allowed the feature table to be constructed with the expected accuracy. As seen here, the window marks the center of each R and T wave. After the two functions were tested and verified on over 1000 events from 15 patients, the algorithms were ready to contribute to the rest of the system. 

<figure>
        <img src="https://github.com/AustinBostwick/ECG-Artificial-Intelligence-Diagnostic/blob/main/BIO431_Pictures/Figure6.png" alt="Figure 6">
</figure>
The results of this image were a very important example of the more advanced algorithms to help feature extraction. As seen in the above window, the area undergoing PVC had a smaller R wave, with a deeper S wave and, most importantly, a large, wide T wave. This T-wave is characteristic of this type of arrhythmia and was a very important feature needed to allow the classifier and network to distinguish this event. All PVC events for multiple patients were tested to ensure the specific detection of these types of T-waves. Analyzing the results of the feature table showed that this function flagged all necessary events but was also showing some higher sensitivity and giving some false positives. This event is addressed in the limitations section. 
The results of the Pan-Tompkins Algorithms allowed us to move on to the feature engineering table construction with confidence that the waves of interest were identified in the majority of cases. The overall finding was that following standard Pan-Tompkins procedures was highly effective for our purpose on the MIT-BIH database.

# Feature Engineering Table
To maximize our training data, we used multiple patients' ECG data. We only used patient data containing annotations to test our neural network's accuracy. This also diversified our data, so there were different total amplitude values, average standard deviations, and abnormalities. The table that we generated contains 21,090 data points. Below is a shortened version of the full spreadsheet with multiple patients and different event types.  

| Patient ID | Beginning Index | Ending Index | Mean_R_Amplitude | STD_R_Amplitude | Average_RR | Variance_RR | Mean_RR_Time | Mean_T_Wave_Amplitude | Variance_TT_Time | STD_T_Wave_Amplitude | Abnormal T Wave Height | Event_Type |
|------------|------------------|--------------|------------------|------------------|------------|-------------|--------------|-----------------------|------------------|----------------------|------------------------|------------|
| 800m | 6009 | 6268 | 0.915 | 0.34828 | 0.93359 | 0.00027 | 1.49218 | 0.07 | 0 | 0.27478 | FALSE | N |
| 800m | 6128 | 6389 | 1.07333 | 0.04195 | 0.94140 | 3.05e-05 | 1.91406 | -0.095 | 0 | 0.02828 | FALSE | N |
| 800m | 6248 | 6484 | 1.02167 | 0.08005 | 0.84375 | 0.0206 | 0.28906 | -0.015 | 0 | 0.28282 | FALSE | N |
| 800m | 6369 | 6632 | 0.99167 | 0.09385 | 0.94922 | 0.0857 | 0.59375 | -0.15167 | 0.65057 | 0.1184 | FALSE | S |
| 800m | 6464 | 6768 | 0.87833 | 0.1025 | 1.10938 | 0.00403 | 0.09375 | -0.455 | 0 | 0.3252 | FALSE | N |
| 801m | 13581 | 13601 | 0.96333 | 0.2291 | 0.69531 | 0.0175 | 0.27083 | 0.0575 | 0.1666 | 0.3395 | FALSE | N |
| 801m | 13591 | 13613 | 1.08833 | 0.3335 | 0.77344 | 0.0590 | 0.625 | 0.31 | 0 | 0.1131 | TRUE | S |
| 801m | 13611 | 13633 | 1.235 | 0.2001 | 0.78125 | 0.0001 | 0.03906 | 0.085 | 0 | 0.6576 | FALSE | N |
| 802m | 14509 | 14731 | 0.37 | 0.3174 | 0.78125 | 0.0825 | 0.0859 | -0.51 | 0 | 0.6929 | FALSE | N |
| 802m | 14634 | 14896 | 0.4487 | 0.3220 | 0.6276 | 0.2179 | 0.0781 | 0.895 | 0 | 0.6858 | FALSE | V |
| 802m | 14711 | 15020 | 0.4612 | 0.3372 | 0.7604 | 0.2511 | 0.9570 | 0.4333 | 1.5449 | 0.9352 | FALSE | N |
| 805m | 17311 | 17334 | 0.1833 | 0.7629 | 0.8125 | 0.0010 | 0.7031 | -0.43 | 0 | 0.1697 | TRUE | N |
| 805m | 17463 | 17477 | 0.7525 | 0.6262 | 0.3125 | 0.0158 | 0.1562 | 1.625 | 0 | 0.6293 | TRUE | V |
| 805m | 17469 | 17488 | 0.7001 | 0.4739 | 0.1015 | 0.1406 | 0.25 | 2.02 | 0 | 1.4142 | TRUE | N |
| 805m | 17586 | 17611 | 0.2766 | 0.6755 | 0.8632 | 3.05e-05 | 0.72656 | -0.32 | 0 | 0.2828 | TRUE | N |
| 805m | 17631 | 17642 | 1.7253 | 0.3906 | 0.125 | 0 | 0.125 | -0.155 | 0 | 0.4596 | TRUE | V |
| 805m | 17665 | 17689 | 0.2466 | 0.8615 | 0.8632 | 3.05e-05 | 0.72656 | -0.425 | 0 | 0.3464 | TRUE | N |
| 805m | 17710 | 17724 | 0.5135 | 0.4648 | 0.0332 | 0.0234 | 0.375 | 2.505 | 0 | 0.6010 | TRUE | V |

## Table: Feature Engineering Table Results Example

# Compact Neural Network Training and Testing Results
<figure>
        <img src="https://github.com/AustinBostwick/ECG-Artificial-Intelligence-Diagnostic/blob/main/BIO431_Pictures/Figure7.png" alt="Figure 7">
</figure>
As seen above, the training took 139 epochs. This large amount of iterations was necessary due to the few neurons in each layer. The weights needed many epochs to be optimized. This architecture did not produce the results we were looking for regarding classification accuracy. This neural network's only advantage was its high training and validation speed. 
<figure>
        <img src="https://github.com/AustinBostwick/ECG-Artificial-Intelligence-Diagnostic/blob/main/BIO431_Pictures/Figure8.png" alt="Figure 7">
</figure>
As seen above, the compact network was highly accurate and sensitive to normal cases. The two arrhythmias of interest to classify were both under 50% success rates. This was far too low for the expected outcome terms, so we designed a more advanced network. 
<figure>
        <img src="https://github.com/AustinBostwick/ECG-Artificial-Intelligence-Diagnostic/blob/main/BIO431_Pictures/Figure9.png" alt="Figure 9">
</figure>
The last six rows show the training loss and validation loss at a steady value. Although the patience was set, it took many epochs to reach this outcome. A redesign was needed for fewer training iterations and higher accuracy classification, as less than 50% on PVC and SP beats was unacceptable. 

# Neural Network Training and Testing Results

<figure>
        <img src="https://github.com/AustinBostwick/ECG-Artificial-Intelligence-Diagnostic/blob/main/BIO431_Pictures/Figure10.png" alt="Figure 10">
</figure>
<figure>
        <img src="https://github.com/AustinBostwick/ECG-Artificial-Intelligence-Diagnostic/blob/main/BIO431_Pictures/Figure11.png" alt="Figure 10">
</figure>
<figure>
        <img src="https://github.com/AustinBostwick/ECG-Artificial-Intelligence-Diagnostic/blob/main/BIO431_Pictures/Figure12.png" alt="Figure 10">
</figure>

The figure above is the result of the architecture seen in Figure 4. We achieved fairly good accuracy on all of the classification groups with normal being the highest at around 98% for training and test. The two other cases had an accuracy of approximately 70-80% for both training and test. Although we were striving for higher accuracy for all cases this was quite a successful result, simultaneously avoiding overfitting. Would this model, as it stands, be accurate enough to be used in a clinical setting? Most likely not. But it shows the correct framework and methodology to develop into a very successful model. 

# Case of Overfitting
<figure>
        <img src="https://github.com/AustinBostwick/ECG-Artificial-Intelligence-Diagnostic/blob/main/BIO431_Pictures/Figure13.png" alt="Figure 13">
</figure>
<figure>
        <img src="https://github.com/AustinBostwick/ECG-Artificial-Intelligence-Diagnostic/blob/main/BIO431_Pictures/Figure14.png" alt="Figure 13">
</figure>
Figures 13 and 14 above, show the results and training when we allowed the model to run for the full epochs. As it can be seen, the accuracy was certainly higher than that of the other results in Figure 12. Still, based on the training validation in Figure 13, it is obvious there is overfitting. To avoid this, we used the “Validation Patience” option in the model to automate the early stopping.
# Limitations
## Pan Tompkins
Generally, the Pan-Tompkins algorithms worked well for the selected MIT-BIH database. However, some limitations of these algorithms become more prominent if training and testing are done using a different database. For the algorithms in general, some of them were thresholded by a single integer. This worked well for the selected database as it had minimal drift in noise in its signal collection. For the abnormal T-wave, the thresholding was based on the scaled average of the signal at that time, making it more dynamic. The team considered dynamic thresholding for all the algorithms to allow for more variance in the input data. It was found that complete dynamic thresholding had limited literature and would have taken much further research and testing. Instead, the team decided that the multiple algorithms that were thresholded for different events would be a practical solution. If this project were extended to other hospital’s databases, it would likely need more algorithms and further research into dynamic thresholding. Better filters for baseline wander and noise could also resolve many of these concerns. 
The main limitation of the Normal T-wave detection was the limit of masking out the R-wave. This was highly effective in most cases, but in arrhythmias, the R-wave is significantly delayed or altered, which could lead to some inaccuracies if the Normal R-wave detection had issues finding the wave to mask out for the T-wave. It was found in this database that this was not a large limiting factor, as most of the arrhythmias included did not alter or delete the R-wave to an unrecognizable amount. This could be resolved by ensuring the windowing is accurate through a dynamic window. 
The main limitation of the abnormal T-wave algorithm was that it was highly sensitive to the PVC event but this meant the specificity was lower, leading to some false positives. Testing and tuning was done to reduce this occurrence, but it was found in testing the neural network that despite some false positives, the true positives were of high importance and increased its ability to detect and classify PVC events. This limitation would be fixed by better filtering and masking of the R-wave and more advanced thresholding to ensure higher specificity. 
## Feature Engineering
This process took the longest to compute. It took about 2-3 minutes to perform on 15 patients. Future work would suggest more data preprocessing to remove windows that were deleted later on to improve the speed of this portion of the system. 
## Compact Neural Network
The compact neural network had many limitations. These limitations were the main reason that we moved on to a more advanced and customizable design. The functions fitcnet and predicted were useful for getting quick results, but a more controlled approach was necessary. It needed a larger hidden layer depth and number of layers to hit the accuracy we wanted. As seen in the results, the accuracy in classifying the two arrhythmias was under 50%. This main limitation was why the team moved on to a more customizable and larger design. 
## Neural Network
The limitations of the full neural network mainly had to do with the network architecture we decided on. With more thought on this project, I think it would’ve been more successful to have done either a Long-term network or a Convolutional neural network because of their higher-use applications in the ECG context. Since we did not use these methods, we were limited in our architecture and development. The network still provided good results, but having a second model of a different methodology may have provided even more accuracy. Nonetheless, the only limitation was how it was trained on our data. We were highly overpopulated with normal cases; thus, the different cases' sensitivity and specificity were affected. We did our best to alleviate this with Validation Patience but were limited in being able to do early stopping to avoid overfitting and still have accuracy, which is very important in this clinical setting. 

# Scope/Conclusion 
The scope of this project encompassed the development and implementation of an advanced neural network model tailored for the classification of electrocardiogram signals to detect arrhythmias. This work involved substantially enhancing the Pan-Tompkins algorithm, precise feature engineering from ECG data, and rigorous neural network training to optimize classification accuracy.
This project required integrating complex signal processing techniques and machine learning algorithms, highlighting the application of comprehensive engineering principles at an advanced level. The project addressed critical challenges in cardiac diagnostics by refining the Pan-Tompkins algorithm to improve its applicability for arrhythmia detection and developing a robust feature engineering table that effectively captures essential characteristics of ECG signals.
Moreover, the implantation of a compact and then a more advanced neural network model demonstrated significant technical competency in handling and analyzing biomedical data. The efforts to optimize these models based on the training outcomes and their potential to support real-time diagnostic decision-making in clinical settings underscore the project's practical relevance and technical depth.
The project’s contribution to enhancing diagnostic accuracy in healthcare, particularly through automated ECG analysis, aligns with the goals of improving patient outcomes and operational efficiencies in medical practices. This is particularly critical in resource-constrained areas, where such automated tools can significantly improve accessibility and quality of care.
In conclusion, the extensive work done in this project far exceeds the scope of just a single homework assignment. Throughout this project, knowledge gained from multiple homework assignments was utilized and expanded upon. Particularly when it comes to using the Pan-Tompkins algorithm as well as the implantation of a complex neural network for ECG signal classification. The integration of these sophisticated techniques demonstrates a deep understanding of the subject matter and significantly contributes to the development of practical solutions in the field of medical diagnostics.

## References
Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, Mietus JE, Moody GB, 

Peng C-K, Stanley HE. PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research 

Resource for Complex Physiologic Signals. Circulation 101(23):e215-e220 [Circulation Electronic Pages; http://circ.ahajournals.org/content/101/23/e215.full]; 2000 (June 13).

Pan, J., & Tompkins, W. J. (1985, March). A Real-Time QRS Detection Algorithm. IEEE Transactions on Biomedical Engineering, 32(3). https://ieeexplore.ieee.org/document/4122029

Liu F, Liu C, Jiang X, Zhang Z, Zhang Y, Li J, Wei S. Performance Analysis of Ten Common QRS 

Detectors on Different ECG Application Cases. J Healthc Eng. (2018, May). Performance Analysis of Ten Common QRS Detectors on Different ECG Application Cases, National Center for Biotechnology Information. Retrieved from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5964584/


