import argparse
from pykeen.pipeline import pipeline

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--embedding_dim', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=3000)
args = parser.parse_args()

whitelist_path = "/ibex/scratch/projects/c2014/alelopezv/2502_AI/kg/F57PRr2/data/whitelist_nocov.txt"
with open(whitelist_path, 'r') as file:
    whitelist = [line.strip() for line in file if line.strip()]

# Train the model
result = pipeline(
    training="data/train_data_new.tsv",
    testing="data/test_data_new.tsv",
    evaluation_relation_whitelist={'has_function'},
    evaluation_entity_whitelist=whitelist,
    model=args.model,
    model_kwargs=dict(embedding_dim=args.embedding_dim),
    training_kwargs=dict(
        num_epochs=100,
        batch_size=args.batch_size,
        use_tqdm_batch=True
    ),
    random_seed=42,
    #how many negative samples are there for one positive 
)

# Save model and report
output_path = f"new_results/{args.model}_100e_{args.batch_size}_e{args.embedding_dim}"
result.save_to_directory(output_path)
print(f"[✓] {args.model} training complete and saved to {output_path}")

#results/whitelist: whitelist(relation),num_epochs=5,batch_size=5000