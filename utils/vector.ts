import {
  PGVectorStore,
  DistanceStrategy,
} from "@langchain/community/vectorstores/pgvector";
import { PoolConfig } from "pg";
import { embeddings } from "./openai";
require("dotenv").config();

export const getVectoreStore = async () => {
  console.log(process.env.DB_HOST)
  const config = {
    postgresConnectionOptions: {
      type: "postgres",
      host: process.env.DB_HOST,
      port: process.env.DB_PORT,
      user: process.env.DB_USER,
      password: process.env.DB_PASSWORD,
      database: process.env.DB_NAME,
    } as PoolConfig,
    tableName: "testlangchainjs",
    columns: {
      idColumnName: "id",
      vectorColumnName: "vector",
      contentColumnName: "content",
      metadataColumnName: "metadata",
    },
    // supported distance strategies: cosine (default), innerProduct, or euclidean
    distanceStrategy: "cosine" as DistanceStrategy,
  };
  const vectorStore = await PGVectorStore.initialize(embeddings, config);
  return vectorStore;
};
