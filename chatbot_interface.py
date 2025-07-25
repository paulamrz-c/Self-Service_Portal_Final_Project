import streamlit as st
import pandas as pd

# Load the CSV file containing resource data
df = pd.read_csv("student_resources_index.csv")

# Set the page title
st.title("🎓 Student Resources Chatbot")

# Display a short instruction
st.write("Ask a question and I’ll suggest relevant student support resources.")

# Text input from the user
user_query = st.text_input("Your question:")

# Only proceed if the user entered something
if user_query:
    # Create a boolean mask to filter rows that contain the keyword
    # in either the Title or Description columns (case-insensitive)
    mask = df["Title"].str.contains(user_query, case=False, na=False) | \
           df["Description"].str.contains(user_query, case=False, na=False)

    # Filter the dataframe
    results = df[mask]

    # Show results if found
    if not results.empty:
        st.success(f"🔍 Found {len(results)} matching resource(s):")

        # Loop through the filtered results and display them
        for _, row in results.iterrows():
            st.markdown(f"### [{row['Title']}]({row['Link']})")
            st.markdown(row['Description'])
            st.markdown("---")
    else:
        # If no match was found
        st.warning("⚠️ No resources found. Try different keywords.")
