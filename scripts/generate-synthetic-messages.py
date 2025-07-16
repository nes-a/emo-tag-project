import pandas as pd
import ollama
import time
import re
import random

# --- Configuration ---
NUM_MESSAGES_PER_EMOTION_TYPE = 1200 # Set this to a higher number (e.g., 250) to reach 10,000+ messages
OLLAMA_MODEL = "llama2" # Or whatever model you are using, e.g., "llama3"

# Your comprehensive list of emotions and emotion pairs
TARGET_EMOTIONS = [
    ['fear'],
    ['excitement'],
    ['desire'],
    ['disgust'],
    ['embarrassment'],
    ['grief'],
    ['nervousness'],
    ['pride'],
    ['relief'],
    ['remorse'],
]

output_file = 'data/synthetic/synthetic-emotion-data.csv'

# --- Conversation Contexts for Mid-Conversation Prompts ---
CONVERSATION_CONTEXTS = [
    "a natural reply in a casual text conversation after someone else has just spoken",
    "a follow-up message reacting to something previously mentioned",
    "a message that would fit in the middle of a chat about a recent event",
    "a short response to a question or statement in an ongoing discussion",
    "a message showing your thoughts as part of a dialogue about a shared experience",
    "a quick, informal message in response to a statement or question",
    "a conversational snippet that assumes prior context",
    "a reaction to a piece of news that someone just shared",
    "a brief remark contributing to an ongoing dialogue",
    "a casual comment in the middle of an informal chat",
    "a simple reply to a previous statement or question",
    "a text message sent to a friend about a personal experience",
    "a message responding to a new piece of information",
    "a thought shared in a free-flowing conversation",
    "a concise reaction to something someone just said",
    "a short message indicating understanding or acknowledgement",
    "a message expressing a quick opinion or feeling",
    "a conversational opening or closing remark",
    "a message that acts as a bridge between two topics in a chat",
    "a simple expression of a feeling after hearing something"
]

# --- Functions ---

def clean_generated_text(text):
    clean_text = text.strip()

    # Remove markdown code blocks
    if clean_text.startswith("```") and clean_text.endswith("```"):
        clean_text = clean_text.strip("`").strip()
        if '\n' in clean_text:
            first_line = clean_text.split('\n')[0].strip()
            if not first_line.startswith('"') and not first_line.endswith('"') and len(first_line.split()) < 3:
                clean_text = clean_text.split('\n', 1)[1].strip()

    # Remove common preambles
    preambles = [
        r"^(Here is a message|Message|Response|Text|Output|Generated text|Here's a message|Sure, here's a message|Here's the message|Here's a possible message|Okay, here's a message|I'm happy to help, here's a message):?\s*",
        r"^\[\"(.*?)\"\]\s*",
        r"^\-\s*",
        r"^\d+\.\s*",
        r"^\s*-\s*",
        r"^\"(.*?)\"\s*",
        r"^\'(.*?)\'\s*"
    ]
    for p in preambles:
        clean_text = re.sub(p, '', clean_text, flags=re.IGNORECASE).strip()

    # If multiple lines, take only the first line/sentence
    if '\n' in clean_text:
        first_line = clean_text.split('\n')[0].strip()
        clean_text = first_line

    # Remove outer quotes
    if (clean_text.startswith('"') and clean_text.endswith('"')) or \
       (clean_text.startswith("'") and clean_text.endswith("'")):
        clean_text = clean_text[1:-1].strip()

    return clean_text


def generate_message_for_emotion(emotion_types, retries=5):
    emotions_str = " and ".join(emotion_types)
    context = random.choice(CONVERSATION_CONTEXTS)

    system_prompt = (
        f"You are an AI assistant specialized in crafting natural and realistic "
        f"text messages that fit a casual conversation context. "
        f"Your goal is to generate short messages (1-3 sentences) that primarily "
        f"express the feeling(s) of {emotions_str}. "
        f"Avoid explicit phrases like 'I feel [emotion]' or 'This makes me [emotion]'. "
        f"Ensure the message feels like a genuine, spontaneous human expression."
    )

    user_prompt = (
        f"Generate a short (1-3 sentences) text message that expresses **{emotions_str}** as {context}. "
        f"The message should be natural and conversational. "
        f"Do not include any preambles, explanations, or quotes around the message. "
        f"Just the message itself."
    )

    for attempt in range(retries):
        try:
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt},
                ]
            )
            raw_message = response['message']['content']
            clean_message = clean_generated_text(raw_message)

            if len(clean_message.strip()) > 5:
                return clean_message
            else:
                print(f"    Attempt {attempt + 1}: Generated message too short or empty for {emotions_str}: '{clean_message}'. Retrying...")
        except Exception as e:
            print(f"    Attempt {attempt + 1}: Error generating message for {emotions_str}: {e}. Retrying...")
        time.sleep(1)

    return None


# --- Main Execution ---
def main():
    start_time = time.time()
    print(f"Starting message generation with model: {OLLAMA_MODEL}")
    print(f"Targeting {NUM_MESSAGES_PER_EMOTION_TYPE} messages per emotion type.")
    print(f"Output file: {output_file}\n")

    all_generated_data = []
    existing_df = pd.DataFrame()
    try:
        existing_df = pd.read_csv(output_file)
        if 'emotion' not in existing_df.columns:
            existing_df['emotion'] = ''
        if 'message' not in existing_df.columns:
            existing_df['message'] = ''
        all_generated_data = existing_df.to_dict('records')
        print(f"Loaded {len(existing_df)} existing messages from {output_file}.")
    except FileNotFoundError:
        print(f"No existing data found at {output_file}. Starting fresh.")
    except Exception as e:
        print(f"Error loading existing data: {e}. Starting fresh.")

    current_counts = {}
    if not existing_df.empty:
        for index, row in existing_df.iterrows():
            emotions_str = str(row['emotion'])
            if '[' in emotions_str and ']' in emotions_str:
                try:
                    parsed_emotions = eval(emotions_str)
                    if isinstance(parsed_emotions, list):
                        key = tuple(sorted(parsed_emotions))
                    else:
                        key = tuple(sorted([emotions_str]))
                except:
                    key = tuple(sorted([emotions_str]))
            else:
                key = tuple(sorted([emotions_str]))
            current_counts[key] = current_counts.get(key, 0) + 1

    print("\nCurrent counts before generation:")
    for key, count in current_counts.items():
        print(f"- {list(key)}: {count}")
    print("-" * 30)


    for i, emotion_type_list in enumerate(TARGET_EMOTIONS):
        emotions_str = " and ".join(emotion_type_list)
        emotion_key = tuple(sorted(emotion_type_list))

        current_count = current_counts.get(emotion_key, 0)
        needed_count = NUM_MESSAGES_PER_EMOTION_TYPE - current_count

        if needed_count <= 0:
            print(f"Skipping '{emotions_str}': Already have {current_count} messages.")
            continue

        print(f"Generating {needed_count} new messages for '{emotions_str}' (Current: {current_count})...")
        generated_count_for_type = 0

        while generated_count_for_type < needed_count:
            message = generate_message_for_emotion(emotion_type_list)
            if message:
                row_data = {'message': message, 'emotion': emotion_type_list}
                if len(emotion_type_list) >= 1:
                    row_data['emotion_label_1'] = emotion_type_list[0]
                else:
                    row_data['emotion_label_1'] = ''

                if len(emotion_type_list) > 1:
                    row_data['emotion_label_2'] = emotion_type_list[1]
                else:
                    row_data['emotion_label_2'] = ''

                all_generated_data.append(row_data)
                generated_count_for_type += 1
                print(f"    Generated: '{message}'")
            else:
                print(f"    Failed to generate a suitable message after retries for {emotions_str}. Moving on.")

            time.sleep(0.5)


    final_df = pd.DataFrame(all_generated_data)

    initial_rows = len(final_df)
    final_df.drop_duplicates(subset=['message'], inplace=True)
    duplicates_removed = initial_rows - len(final_df)
    if duplicates_removed > 0:
        print(f"Removed {duplicates_removed} duplicate messages.")

    final_df = final_df.sort_values(by=['emotion', 'message']).reset_index(drop=True)

    final_df.to_csv(output_file, index=False)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nGeneration complete!")
    print(f"Total messages generated (including existing): {len(final_df)}")
    print(f"Total time taken: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()