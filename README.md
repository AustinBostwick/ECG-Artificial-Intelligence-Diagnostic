<h1 style="text-align:center;">Neural Network Models for Electrocardiogram Classification</h1>
<h2 style="text-align:center;">Final Project Report</h2>
<h2 style="text-align:center;"> Austin Bostwick, Nicholas Emer, Jack Lattanzi, and Nate Peebles</h2>

# Abstract (Motivation/Relevance)
Atrial and ventricular fibrillations are well-known risk factors for stroke due to the potential formation of blood clots in the heart that can travel to the brain. Detecting heart arrhythmias early before they devolve into fibrillation through an ECG can lead to the initiation of anticoagulant therapy, which significantly reduces the risk of stroke. Effective ventricular and atrial arrhythmia management can significantly improve a patient's quality of life. It can reduce symptoms such as palpitations, shortness of breath, and fatigue, allowing individuals to maintain a more active and fulfilling lifestyle.
Detection of atrial and ventricular arrhythmias involves analyzing an ECG signal. This process typically requires a highly trained cardiologist who interprets the ECG readings to identify abnormalities. Upon detecting an irregularity, the cardiologist assesses its origin and formulates a treatment strategy tailored to the abnormality's severity and the patient's overall health condition. This can be a time-consuming process, where developing a classification neural network could help streamline the diagnosis process. 
Implementing a neural network for analyzing ECG data could revolutionize real-time diagnostic support, especially in emergencies where quick decision-making is essential. Such automated diagnosis could be invaluable in regions with limited access to cardiologists, enhancing healthcare by delivering quality diagnostics without necessitating an on-site specialist. A remote specialist could receive the preliminary diagnosis made by the neural network and confirm its diagnosis by investigating the ECG segment of interest. The time of diagnosis would be decreased because the classification neural network could send packaged sections of an ECG waveform to a remote cardiologist to make a final diagnosis. Moreover, this could contribute to cost-efficiency within the healthcare system and enhance early detection rates, thereby preventing more significant health issues. 

# Scientific Context
The Pan-Tompkins algorithm is the most widely used QRS complex detector in industry. There are other algorithms based on Pan-Tompkins, such as the Hamilton-Median. There are other algorithms in general, such as RS Slope, Sixth power, “jqrs”, and many more. While a more complicated algorithm may produce better results, the Pan-Tompkins algorithm can be designed quickly and with the same results as those other algorithms. 
At the same time, a fully connected feed-forward neural network was used in the system's design. This is a one-directional information-flowing network that only flows forward with no loops or cycles. This is common for classification networks, which can classify photos or data. This is different from Recurrent Neural Networks, which have loops or reuse data from themselves. The feed-forward network is more suited for predicting a single output, which is what is being done with this project. 


<figure>
        <img src="https://raw.githubusercontent.com/AustinBostwick/" alt="FlowChart">
        <figcaption>Figure 17: Non-Ideal Single-Stage Switch Capacitor Amplifier</figcaption>
</figure>
