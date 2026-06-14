# Understanding the Synthetic Data Generation in the Medical Equipment Project

## Overview
In this project, you transitioned from using static device-level data to a time-series-based approach for predicting medical equipment failures. To achieve this, a script (`scripts/generate_synthetic_iot.py`) was used to generate **synthetic time-series IoT data**.

Here is a detailed explanation of **what** was done to create this new data and **why** it was necessary.

---

## 1. What Did You Do to Create the New Data?

The `generate_synthetic_iot.py` script takes original static equipment datasets (such as the `SmouhaMedicalCenter_cleaned.csv` equipment inventory and `Medical_Device_Failure_dataset.csv`) and simulates daily sensor logs for each device over a specified period (e.g., 365 days). 

Here is exactly how the data generation process works:

### A. Extracting Base Characteristics
The process begins by extracting static information from real-world hospital equipment inventories, such as the `SmouhaMedicalCenter_cleaned.csv` file. This dataset provides a realistic foundation, offering details like hospital department, device names (e.g., monitors, artificial respiration, dialysis machines), purchase years, and technical competence. Key attributes extracted include:
- **Device Type:** Establishes baseline sensor readings (e.g., a Dialysis Machine has different baseline vibrations than an Infusion Pump).
- **Age / Device Life:** Derived from the purchase year (e.g., 2015 vs 2023) to calculate a "wear" factor. Older machines have higher baseline sensor readings and are more prone to fluctuations.
- **Failure Event Count / Health Status:** Used to schedule how many times the device will fail over the simulated period, informed by the device's technical competence and operational life.

### B. Simulating Daily Sensor Readings
For each day in the simulation, the script generates readings for three critical sensors:
1. **Temperature Variance**
2. **Motor Vibration (Hz)**
3. **Voltage Drop**

### C. Introducing Degradation Patterns (Approaching Failure)
The most crucial part of the script is how it simulates machine degradation:
- **Scheduled Failures:** The script schedules failures randomly throughout the year based on the device's failure count.
- **Pre-Failure Spikes:** As a scheduled failure approaches (within 10 days), the script artificially inflates the sensor readings. Temperature and vibration start to spike, and voltage drops become more frequent and severe. 

### D. Adding Realism (Noise and False Flags)
Real-world data is rarely perfect. To make the synthetic data realistic, the script introduces:
- **Random Noise:** Slight daily fluctuations in sensor readings.
- **False Flags:** Occasional one-day anomalies (e.g., a random spike in vibration) that happen far away from actual failures. This forces the Machine Learning model to distinguish between a harmless glitch and actual equipment degradation.

### E. Generating Target Labels
Finally, the script calculates the target variables needed for predictive maintenance models:
- **RUL (Remaining Useful Life):** The exact number of hours and days until the next scheduled failure.
- **Will_Fail_In_72_Hours:** A binary label (0 or 1) indicating if the machine will fail within the next 3 days.

---

## 2. Why Was This Done? (The Rationale)

Creating this synthetic dataset was a critical step for your project for several reasons:

### A. Static Data is Insufficient for Predictive Maintenance
The original dataset (`SmouhaMedicalCenter_cleaned.csv`) only provided a static inventory and summary of the equipment (e.g., "This 2015 monitor has a 65% technical competence"). While this is excellent for basic asset management and statistics, it does not tell you **how** the machine behaves right before it breaks. Predictive maintenance requires understanding the *journey* to failure (the real-time sensor fluctuations), not just the current static state.

### B. Enabling Advanced Time-Series Models (LSTM)
You are using advanced models like **Long Short-Term Memory (LSTM)** neural networks (as seen in `training_lstm.ipynb` and `predict_lstm.py`). LSTMs are designed to look at sequences of data over time to find hidden degradation patterns. By simulating daily logs, you provided the exact type of 3D sequential data (Samples × Time Steps × Features) that LSTMs require to train effectively.

### C. Bridging the Data Availability Gap
High-frequency, real-world IoT sensor data for medical equipment is extremely rare and often heavily guarded due to proprietary or privacy reasons. By using expert-defined baselines (e.g., knowing that an older machine vibrates more before failing), you successfully bridged the gap, creating a highly realistic dataset that serves as a perfect proxy for real-world IoT data.

### D. Formulating a Clear Business Problem
By calculating the `Will_Fail_In_72_Hours` label, you transformed a generic data problem into a highly actionable business solution. In a real hospital setting, predicting a failure 3 days in advance allows maintenance teams to fix a life-saving device *before* it breaks down during a critical procedure.
