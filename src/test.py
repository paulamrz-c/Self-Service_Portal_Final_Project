from retriever import answer




"""
“How do I pay my fees?”

“What is VMock?”

“Where can I find my timetable?”

“What’s the deadline for OSAP?”
"""

print(answer("How can I pay my fees?", model_type="w2v"))
print(answer("How can I pay my fees?", model_type="glove"))
print("*********")

print(answer("What is VMock?", model_type="w2v"))
print(answer("What is VMock?", model_type="glove"))

print(answer("Where can I fix my CV?", model_type="w2v"))
print(answer("Where can I fix my CV?", model_type="glove"))