"""Train the vectorizer."""
import argparse
from pathlib import Path

from sentence_transformers import evaluation, InputExample, losses, SentenceTransformer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from rushifteval_utils import AnnotatedWordPair, load_annotated_data, load_jsonl_vectors, parse_path


def train_vectorizer(annotated_words: list[AnnotatedWordPair], output_path: str | Path) -> None:
    """
    Train the vectorizer.

    :param annotated_words: A list of pairs of word usages and their definitions.
    :param output_path: The path to where the model will be saved.
    """
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

    train_examples = [
        InputExample(texts=[pair.definition1, pair.definition2], label=(pair.mean - 1) / 3) for pair
        in annotated_words]

    train_dataset, val_dataset = train_test_split(train_examples, test_size=0.2,
                                                  random_state=42)

    train_dataloader = DataLoader(train_dataset, batch_size=32)

    train_loss = losses.CosineSimilarityLoss(model)

    evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(val_dataset,
                                                                            name='rus-shift-eval')

    if not Path(output_path).parent.exists():
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=evaluator,
              epochs=3,
              warmup_steps=100,
              evaluation_steps=50,
              output_path=output_path)

    print(f"Training complete. Model saved to {output_path}")


def main():
    """Fine-tune a vectorizer model using rusemshift dataset."""
    parser = argparse.ArgumentParser(
        description="Fine-tune a vectorizer model using rusemshift dataset.")
    parser.add_argument("--tsv", type=str,
                        help="Path to the rusemshift annotations in .tsv format.")
    parser.add_argument("--jsonl", type=str,
                        help="Path to the rusemshift predictions in .jsonl format.")
    parser.add_argument("--output-path", type=str,
                        default="models/paraphrase-multilingual-mpnet-base-dm",
                        help="Path to where the fine-tuned model will be.")
    args = parser.parse_args()

    jsonl_file_path = parse_path(args.jsonl)
    tsv_file_path = parse_path(args.tsv)
    output_path = parse_path(args.output_path)

    jsonl_vectors = load_jsonl_vectors(jsonl_file_path)
    annotated_words = load_annotated_data(tsv_file_path, jsonl_vectors)

    train_vectorizer(annotated_words, output_path)


if __name__ == '__main__':
    main()
