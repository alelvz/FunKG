from pykeen.models import predict_links
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline_result_from_directory

# Load result and training factory
result = pipeline_result_from_directory("results/kg_model")
model = result.model
triples_factory = result.training

# Predict all missing has_function links
df_predictions = predict_links(
    model=model,
    head_label_all_filtered=True,
    relation_label="has_function",
    triples_factory=triples_factory
)

# Save top predictions
df = df_predictions.get_prediction_df()
df.to_csv("results/predicted_links.tsv", sep="\t", index=False)
print("[✓] Predictions saved to results/predicted_links.tsv")
