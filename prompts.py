CAPTION_PROMPT = """
You will be given an image and a question pair, along with the original answer and the new answer the image needs to be modified to.

Your job is to generate a text-to-image prompt that can be used with a diffusion model. Based on the question and answer, write a detailed caption so that all necessary details are included and the question remains solvable.

Additionally, modify the image so that the correct answer changes. For example, if the question asks “How many people are in the image?”, change the description of the image to have more people.

Here is the original question: {question} and the original answer: {original_answer}. The new answer should now become {new_answer}. Do not output any other text than the caption itself.
"""

FILTER_PROMPT = """
You will be given a question and an image pair, along with the answer. Your job is to critically analyze the image-question pair to verify that the question can be correctly answered.

In particular, ensure that one can deduce the correct answer choice and that choice only. If there is any ambiguity, you must reject this question. When finalizing your decision, do NOT take into consideration the quality of the image. As long as the question remains solvable, you should keep it. You can reason about your given task for as long as you want, but provide your answer in the following format: "Answer: ANSWER" and answer with KEEP or REJECT.

Here is the question: {question} and the answer is {answer}.
"""