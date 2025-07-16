import { StringOutputParser } from "@langchain/core/output_parsers";
import {
  RunnablePassthrough,
  RunnableSequence,
  RunnableWithMessageHistory,
} from "@langchain/core/runnables";
import { ChatOpenAI } from "@langchain/openai";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import pg from "pg";
import { Document } from "@langchain/core/documents";
import { getVectoreStore } from "./vector";
import { PostgresChatMessageHistory } from "@langchain/community/stores/message/postgres";
require("dotenv").config();

const GREETINGS = [
  "bonjour", "bonsoir", "salut", "slt", "hello", "hi", "hey", "yoo", "salam", "slm",
  "salem", "salam alaykoum", "salam aleykoum", "salem alikoum", "salam alikoum",
  "azul", "allo", "yo", "good morning", "good evening", "morning", "evening",
];

function containsGreeting(text: string): boolean {
  const lowered = text.toLowerCase();
  return GREETINGS.some((greet) => lowered.includes(greet));
}

function isOnlyGreeting(text: string): boolean {
  const lowered = text.toLowerCase().trim();
  return GREETINGS.some((greet) => {
    const regex = new RegExp(`^${greet}[\\s!.,?]*$`, "i");
    return regex.test(lowered);
  });
}

export const mainFunction = async (userInput: string, phoneNumber: string) => {
  console.log("User input:", userInput);
  console.log("Phone number/sessionId:", phoneNumber);

  // Case 1: Greeting only
  if (isOnlyGreeting(userInput)) {
    console.log("Detected a greeting message.");
    return `Bonjour! Je suis l'assistant virtuel de l'Université Internationale de Rabat. Comment puis-je vous aider aujourd'hui ? Avez-vous des questions sur nos programmes, les admissions ou peut-être cherchez-vous des informations générales sur l'université ?`;
  }

  // Vector store and retriever setup
  console.log("Initializing vector store...");
  const vectorStore = await getVectoreStore();
  const retriever = vectorStore.asRetriever();
  console.log("Vector store initialized.");

  const convertDocsToString = (documents: Document[]): string => {
    console.log("Converting retrieved documents to string...");
    return documents
      .map((document) => `<doc>\n${document.pageContent}\n</doc>`)
      .join("\n");
  };

  const REPHRASE_QUESTION_SYSTEM_TEMPLATE = `Vous êtes l'assistant virtuel de l'Université Internationale de Rabat.

Sinon, reformulez la question comme une question autonome sans mentionner la conversation précédente.`;

  const rephraseQuestionChainPrompt = ChatPromptTemplate.fromMessages([
    ["system", REPHRASE_QUESTION_SYSTEM_TEMPLATE],
    new MessagesPlaceholder("history"),
    ["human", "Rephrase the following question as a standalone question:\n{question}"],
  ]);

  const rephraseQuestionChain = RunnableSequence.from([
    rephraseQuestionChainPrompt,
    new ChatOpenAI({ temperature: 0.1, modelName: "gpt-3.5-turbo-1106" }),
    new StringOutputParser(),
  ]);

  const ANSWER_CHAIN_SYSTEM_TEMPLATE = `Vous êtes l'assistant virtuel de l'Université Internationale de Rabat. Répondez poliment, professionnellement, et uniquement à partir des informations fournies dans le contexte ci-dessous.

Si vous ne trouvez pas l'information dans le contexte, dites simplement :
"Pouvez-vous reformuler cette question ?"

<context>
{context}
</context>`;

  const answerGenerationChainPrompt = ChatPromptTemplate.fromMessages([
    ["system", ANSWER_CHAIN_SYSTEM_TEMPLATE],
    new MessagesPlaceholder("history"),
    ["human", "Now, answer this question using the previous context and chat history:\n{standalone_question}"],
  ]);

  const documentRetrievalChain = RunnableSequence.from([
    (input) => {
      console.log("Retrieving documents for:", input);
      return input.standalone_question;
    },
    retriever,
    (docs: Document[]) => {
      console.log(`Retrieved ${docs.length} documents.`);
      return convertDocsToString(docs);
    },
  ]);

  const conversationalRetrievalChain = RunnableSequence.from([
    RunnablePassthrough.assign({
      standalone_question: async (input) => {
        const rephrased = await rephraseQuestionChain.invoke({ question: input.question });
        console.log("Standalone question:", rephrased);
        return rephrased;
      },
    }),
    RunnablePassthrough.assign({
      context: async (input) => {
        const context = await documentRetrievalChain.invoke(input);
        console.log("Retrieved context:", context.substring(0, 500), "...");
        return context;
      },
    }),
    answerGenerationChainPrompt,
    new ChatOpenAI({ modelName: "gpt-3.5-turbo" }),
    new StringOutputParser(),
  ]);

  const poolConfig = {
    host: process.env.DB_HOST,
    port: Number(process.env.DB_PORT),
    user: process.env.DB_USER,
    password: process.env.DB_PASSWORD,
    database: process.env.DB_NAME,
  };

  console.log("Initializing PostgreSQL pool...");
  const pool = new pg.Pool(poolConfig);

  const finalRetrievalChain = new RunnableWithMessageHistory({
    runnable: conversationalRetrievalChain,
    getMessageHistory: async (sessionId) => {
      console.log("Fetching message history for session:", sessionId);
      return new PostgresChatMessageHistory({ sessionId, pool });
    },
    historyMessagesKey: "history",
    inputMessagesKey: "question",
  });

  console.log("Invoking the final chain...");
  const finalResult = await finalRetrievalChain.invoke(
    { question: userInput },
    { configurable: { sessionId: phoneNumber } }
  );

  console.log("Final result:", finalResult);
  return finalResult;
};
