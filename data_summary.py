import matplotlib.pyplot as plt

# Re-entering the updated data for each participant
participants = ["Participant 1", "Participant 2", "Participant 3", "Participant 4", 
                "Participant 5", "Participant 6", "Participant 7", "Participant 8", 
                "Participant 9", "Participant 10"]
updated_undetected_counts = [8, 3, 9, 11, 7, 10, 6, 5, 12, 4]

# Creating the bar chart without the word 'Blue' in the title
plt.figure(figsize=(10, 6))
plt.bar(participants, updated_undetected_counts, color='blue')

# Adding titles and labels
plt.title("Updated Number of Undetected Trials per Participant")
plt.xlabel("Participants")
plt.ylabel("Number of Undetected Trials")
plt.xticks(rotation=45)

# Show the plot
plt.show()
