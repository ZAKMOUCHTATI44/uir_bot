import { Annotation, StateGraph } from "@langchain/langgraph";
import { llm } from "./openai";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import { Document } from "@langchain/core/documents";
import { getVectoreStore } from "./vector";
import {
  RunnableSequence,
  RunnableWithMessageHistory,
} from "@langchain/core/runnables";
import { PostgresChatMessageHistory } from "@langchain/community/stores/message/postgres";
import pg from "pg";
import { StringOutputParser } from "@langchain/core/output_parsers";

export const genrateAnswer = async (userInput: string, sessionId: string) => {
  const vectorStore = await getVectoreStore();

  const template = `Vous êtes l'assistant de l'Université Internationale de Rabat, un chercheur expérimenté, expert dans l'interprétation et la réponse aux questions basées sur des sources fournies.

En utilisant uniquement le contexte fourni, vous devez répondre à la question de l'utilisateur au mieux de vos capacités, sans jamais vous appuyer sur des connaissances extérieures.

Votre réponse doit être très détaillée, explicite et pédagogique.
et répondre avec la meme langue que prompt

{context}

Question: {question}

Helpful Answer:`;

  const promptTemplate = ChatPromptTemplate.fromMessages([["user", template]]);

  // Define a single state type for the graph
  interface GraphState {
    question: string;
    context?: Document[];
    answer?: string;
  }

  const StateAnnotation = Annotation.Root({
    question: Annotation<string>,
    context: Annotation<Document[]>,
    answer: Annotation<string>,
  });

  const retrieve = async (state: GraphState) => {
    const retrievedDocs = await vectorStore.similaritySearch(state.question);
    return { context: retrievedDocs };
  };

  const generate = async (state: GraphState) => {
    if (!state.context) throw new Error("No context provided");
    const docsContent = state.context.map((doc) => doc.pageContent).join("\n");
    const messages = await promptTemplate.formatMessages({
      question: state.question,
      context: docsContent,
    });

    const response = await llm.invoke(messages);
    if (!response.content) {
      throw new Error("No valid response content found from LLM");
    }

    return { answer: response.content };
  };

  const graph = new StateGraph(StateAnnotation)
    .addNode("retrieve", retrieve)
    .addNode("generate", generate)
    .addEdge("__start__", "retrieve")
    .addEdge("retrieve", "generate")
    .addEdge("generate", "__end__")
    .compile();

  const poolConfig = {
    host: process.env.DB_HOST,
    port: Number(process.env.DB_PORT),
    user: process.env.DB_USER,
    password: process.env.DB_PASSWORD,
    database: process.env.DB_NAME,
  };

  const pool = new pg.Pool(poolConfig);

  const prompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      "You are a helpful assistant. Answer all questions to the best of your ability.",
    ],
    new MessagesPlaceholder("chat_history"),
    ["human", "{input}"],
  ]);

  const chain = prompt.pipe(llm).pipe(new StringOutputParser());

  const composedChainWithLambda = RunnableSequence.from([
    chain,
    (input) => ({ joke: input }),
    prompt,
    llm,
    new StringOutputParser(),
  ]);

  const memoryRunnable = new RunnableWithMessageHistory({
    runnable: composedChainWithLambda,
    inputMessagesKey: "input",
    historyMessagesKey: "chat_history",
    getMessageHistory: async (sessionId) => {
      const chatHistory = new PostgresChatMessageHistory({
        sessionId,
        pool,
      });
      console.log(chatHistory);
      return chatHistory;
    },
  });

  try {
    const result = await memoryRunnable.invoke(
      { input: userInput },
      { configurable: { sessionId } }
    );
    return result;
  } catch (error) {
    console.error("Error generating answer:", error);
    throw error;
  } finally {
    await pool.end();
  }
};
