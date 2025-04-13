from pinecone import Pinecone

pc = Pinecone(
    api_key="pcsk_HjLET_2TLGTcZheZRPKUcGFGHJQ3vvDtA8m1CBjYA6Fj8UexQ2XnbGNJSDk38K53BmSfV"
)

# To get the unique host for an index,
# see https://docs.pinecone.io/guides/data/target-an-index
index = pc.Index(host="https://owlai-law-sq85boh.svc.apu-57e2-42f6.pinecone.io")

results = index.fetch(ids=["doc-103"])

print(results)
