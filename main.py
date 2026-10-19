
import os
from datetime import datetime
from google import genai

MODEL = "gemini-3.8-flash"
RESULTS_FILE = "results.txt"

PROMPTS = {
    1: {
        "name": "Zero-Shot Prompting",
        "style": "Zero-shot",
        "prompt": (
            "Explain Artificial Intelligence in simple English "
            "for a college student. Give a definition and 3 examples."
        ),
    },
    2: {
        "name": "Few-Shot Prompting",
        "style": "Few-shot",
        "prompt": """
Classify each review as Positive, Negative, or Neutral.

Review: "The product is excellent!"
Sentiment: Positive

Review: "The service was terrible."
Sentiment: Negative

Review: "The package arrived today."
Sentiment: Neutral

Now classify:
Review: "The application is very useful and easy to use."
Sentiment:
""",
    },
    3: {
        "name": "Reasoning-Style Prompting",
        "style": "Reasoning",
        "prompt": """
A student studies 2 hours on Monday, 3 hours on Tuesday,
and 4 hours on Wednesday.
Calculate the total and average study time.
Show the calculation steps and final answer.
""",
    },
    4: {
        "name": "Text Summarization",
        "style": "Instruction-based",
        "prompt": """
Summarize this paragraph in exactly 3 bullet points:
"Artificial Intelligence enables computers to perform tasks
that normally require human intelligence. Machine learning
allows systems to learn patterns from data. AI is used in
healthcare, education, transportation, and agriculture.
Responsible development is important for privacy, fairness,
and safety."
""",
    },
    5: {
        "name": "Translation",
        "style": "Instruction-based",
        "prompt": """
Translate into Telugu:
"Technology can make education more accessible to everyone."
Give the Telugu translation and a simple English meaning.
""",
    },
    6: {
        "name": "Text Classification",
        "style": "Classification",
        "prompt": """
Classify this message as SPAM or NOT SPAM.
Message: "Congratulations! You have won a prize.
Click here to claim your reward now."

Return:
Category:
Reason:
""",
    },
    7: {
        "name": "Creative Writing",
        "style": "Creative",
        "prompt": """
Write a motivational story of about 150 words about a
college student who learns Python and builds a useful project.
Include a title and positive ending.
""",
    },
    8: {
        "name": "Code Generation",
        "style": "Code generation",
        "prompt": """
Write a Python function to check whether a number is prime.
Include comments, 3 test cases, and a brief explanation.
""",
    },
    9: {
        "name": "Structured Output",
        "style": "Structured output",
        "prompt": """
Return only a valid JSON object about a fictional student
with these fields: name, course, skills (list), beginner_project.
Do not include Markdown formatting.
""",
    },
    10: {
        "name": "Role-Based Prompting",
        "style": "Role-based",
        "prompt": """
Act as a friendly Python tutor for a college beginner.
Explain Python lists using simple language.
Include one code example and three key points.
""",
    },
}


def get_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("\nAPI key not found!")
        print('Set it with: $env:GEMINI_API_KEY="YOUR_API_KEY"')
        return None
    return genai.Client(api_key=api_key)


def generate_response(client, prompt):
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
    )
    return response.text or "No text response was returned."


def save_result(title, prompt, response):
    with open(RESULTS_FILE, "a", encoding="utf-8") as file:
        file.write("\n" + "=" * 65 + "\n")
        file.write(f"Date: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        file.write(f"Experiment: {title}\n")
        file.write("-" * 65 + "\n")
        file.write("PROMPT:\n" + prompt.strip() + "\n\n")
        file.write("MODEL RESPONSE:\n" + response.strip() + "\n")


def run_experiment(client, number):
    experiment = PROMPTS[number]
    print("\n" + "=" * 65)
    print(f"{experiment['name']} ({experiment['style']})")
    print("=" * 65)
    print("Generating response... Please wait.\n")

    try:
        answer = generate_response(client, experiment["prompt"])
        print(answer)
        save_result(experiment["name"], experiment["prompt"], answer)
        print(f"\nSaved to {RESULTS_FILE}")
    except Exception as error:
        print(f"\nRequest failed: {error}")
        print("Check your internet, API key, model availability, and quota.")


def compare_prompts(client):
    task = """
A student studies for 2 hours on Monday, 3 hours on Tuesday,
and 4 hours on Wednesday. Find the total and average study time.
"""

    prompts = {
        "Zero-Shot": task,
        "Few-Shot": """
Use the examples to solve the next problem.

Example:
Hours: 2, 4
Total: 6 hours
Average: 3 hours

Now solve:
A student studies 2 hours on Monday, 3 hours on Tuesday,
and 4 hours on Wednesday. Find the total and average.
""",
        "Reasoning-Style": """
Solve carefully and show the calculations.
A student studies 2 hours on Monday, 3 hours on Tuesday,
and 4 hours on Wednesday.
Calculate total study time and average per day.
""",
    }

    print("\nTHREE PROMPTING STYLES COMPARISON")
    print("=" * 65)

    try:
        for style, prompt in prompts.items():
            print(f"\n{style.upper()} OUTPUT")
            print("-" * 65)
            answer = generate_response(client, prompt)
            print(answer)
            save_result(f"{style} Comparison", prompt, answer)

        print("\nCOMPARISON SUMMARY")
        print("Zero-shot: No examples are provided.")
        print("Few-shot: Examples guide the expected response.")
        print("Reasoning-style: The prompt requests calculation steps.")
        print("Review correctness, clarity, and output format.")
        print(f"\nAll results saved to {RESULTS_FILE}")

    except Exception as error:
        print(f"\nComparison failed: {error}")


def show_menu():
    print("\n" + "=" * 65)
    print("       PROMPT ENGINEERING SHOWCASE")
    print("       Task 1 - Generative AI Internship")
    print("=" * 65)

    for number, item in PROMPTS.items():
        print(f"{number:2}. {item['name']}")

    print("11. Compare three prompting styles")
    print(" 0. Exit")
    print("-" * 65)


def main():
    print("Initializing Prompt Engineering Showcase...")
    client = get_client()
    if client is None:
        return

    while True:
        show_menu()
        choice = input("Choose an option (0-11): ").strip()

        if choice == "0":
            print("\nThank you for using the Prompt Engineering Showcase!")
            break
        elif choice == "11":
            compare_prompts(client)
        elif choice.isdigit() and int(choice) in PROMPTS:
            run_experiment(client, int(choice))
        else:
            print("Invalid choice. Please enter a number from 0 to 11.")


if __name__ == "__main__":
    main()
