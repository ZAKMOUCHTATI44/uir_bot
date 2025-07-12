import { AIMessage, HumanMessage } from "@langchain/core/messages";
import { StringOutputParser } from "@langchain/core/output_parsers";
import {
  RunnablePassthrough,
  RunnableSequence,
  RunnableWithMessageHistory,
} from "@langchain/core/runnables";
import { ChatOpenAI } from "@langchain/openai";
import { ChatMessageHistory } from "langchain/stores/message/in_memory";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import pg from "pg";
import { Document } from "@langchain/core/documents";
import { getVectoreStore } from "./vector";
import { PostgresChatMessageHistory } from "@langchain/community/stores/message/postgres";
require("dotenv").config();

export const mainFunction = async (userInput: string, phoneNumber: string) => {
  const REPHRASE_QUESTION_SYSTEM_TEMPLATE = `Given the conversation and a follow-up question, rewrite it as a standalone question.
If the conversation has no useful context or is empty, return the original question as-is.
If the question contains "bonjour", "hello", "salam", or similar greetings, do NOT rephrase and return the original question.`;

  const vectorStore = await getVectoreStore();
  const retriever = vectorStore.asRetriever();

  const convertDocsToString = (documents: Document[]): string => {
    return documents
      .map((document) => `<doc>\n${document.pageContent}\n</doc>`)
      .join("\n");
  };

  const rephraseQuestionChainPrompt = ChatPromptTemplate.fromMessages([
    ["system", REPHRASE_QUESTION_SYSTEM_TEMPLATE],
    new MessagesPlaceholder("history"),
    [
      "human",
      "Rephrase the following question as a standalone question:\n{question}",
    ],
  ]);

  const rephraseQuestionChain = RunnableSequence.from([
    rephraseQuestionChainPrompt,
    new ChatOpenAI({ temperature: 0.1, modelName: "gpt-3.5-turbo-1106" }),
    new StringOutputParser(),
  ]);

  const ANSWER_CHAIN_SYSTEM_TEMPLATE = `Vous êtes l'assistant virtuel de l'Université Internationale de Rabat – Technopolis Rabat-Shore Rocade Rabat-Salé. Répondez poliment, professionnellement et dans la même langue que la question de l'utilisateur.

Si la question est en français, répondez en français.  
Si elle est en anglais, répondez en anglais.  
Si vous n'arrivez pas à identifier la langue, répondez par défaut en français.

Si la question contient des mots comme "bonjour", "hello", "salam", etc., commencez par :
"Bonjour ! Je suis l'assistant virtuel de l'Université Internationale de Rabat. Comment puis-je vous aider aujourd'hui ?"

Sinon, répondez uniquement au contenu de la question sans ajouter d'informations non présentes dans les documents. Soyez clair, concis et basé sur le contexte fourni.

<context>
{context}
</context>`;

  const answerGenerationChainPrompt = ChatPromptTemplate.fromMessages([
    ["system", ANSWER_CHAIN_SYSTEM_TEMPLATE],
    new MessagesPlaceholder("history"),
    [
      "human",
      "Now, answer this question using the previous context and chat history:\n{standalone_question}",
    ],
  ]);

  const documentRetrievalChain = RunnableSequence.from([
    (input) => input.standalone_question,
    retriever,
    convertDocsToString,
  ]);

  const conversationalRetrievalChain = RunnableSequence.from([
    RunnablePassthrough.assign({
      standalone_question: rephraseQuestionChain,
    }),
    RunnablePassthrough.assign({
      context: documentRetrievalChain,
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

  const pool = new pg.Pool(poolConfig);

  const finalRetrievalChain = new RunnableWithMessageHistory({
    runnable: conversationalRetrievalChain,
    getMessageHistory: async (sessionId) => {
      const chatHistory = new PostgresChatMessageHistory({
        sessionId,
        pool,
      });
      return chatHistory;
    },
    historyMessagesKey: "history",
    inputMessagesKey: "question",
  });

  const finalResult = await finalRetrievalChain.invoke(
    {
      question: userInput,
    },
    {
      configurable: { sessionId: phoneNumber },
    }
  );

  return finalResult;
};
