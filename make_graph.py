import matplotlib.pyplot as plt

# Data definition
participants = range(1, 21)

# Each participant's Cursor Movement success rate (points obtained out of 10), converted to percentage
cursor_movement_accuracy = [
    1*100, 1*100, 1*100, 1*100, 1*100, 1*100, 1*100, 1*100, 1*100, 1*100, 1*100, 1*100, 1*100, 1*100, 1*100, 1*100, 1*100, 1*100, 1*100, 1*100
]

# Plotting the bar graph for Cursor Movement Accuracy
plt.figure(figsize=(10, 6))
plt.bar(participants, cursor_movement_accuracy, color='blue')  # Using 'blue' color as per your reference
plt.title('Scrolling Down Accuracy')
plt.xlabel('Participants')
plt.ylabel('Accuracy[%]')
plt.ylim(0, 100)  # Setting the range for Y-axis to 0-100% for accuracy rate
plt.xticks(participants)
plt.show()
