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
  //   const REPHRASE_QUESTION_SYSTEM_TEMPLATE = `Given the conversation and a follow-up question, rewrite it as a standalone question.
  // If the conversation has no useful context or is empty, return the original question as-is.
  // If the question contains "bonjour", "hello", "salam", or similar greetings, do NOT rephrase and return the original question.`;

  const vectorStore = await getVectoreStore();
  const retriever = vectorStore.asRetriever();

  const convertDocsToString = (documents: Document[]): string => {
    return documents
      .map((document) => `<doc>\n${document.pageContent}\n</doc>`)
      .join("\n");
  };
  const REPHRASE_QUESTION_SYSTEM_TEMPLATE = `
  Vous êtes l'assistant virtuel de l'Université Internationale de Rabat. 
  Votre tâche est de reformuler toute question utilisateur comme une question autonome complète, 
  même si elle dépend du contexte précédent. 
  La question reformulée doit toujours être en français, même si l'utilisateur utilise une autre langue.
  Si vous ne pouvez pas reformuler correctement la question de manière autonome, retournez simplement le dernier message de l'utilisateur tel quel, sans modification.
  `;

  const rephraseQuestionChainPrompt = ChatPromptTemplate.fromMessages([
    ["system", REPHRASE_QUESTION_SYSTEM_TEMPLATE],
    new MessagesPlaceholder("history"),
    [
      "human",
      "Reformule la question suivante comme une question autonome :\n{question}",
    ],
  ]);

  const rephraseQuestionChain = RunnableSequence.from([
    rephraseQuestionChainPrompt,
    new ChatOpenAI({ temperature: 0.1, modelName: "gpt-3.5-turbo-1106" }),
    new StringOutputParser(),
  ]);

  const ANSWER_CHAIN_SYSTEM_TEMPLATE = `Vous êtes l'assistant virtuel de l'Université Internationale de Rabat. Répondez poliment, professionnellement, et dans la même langue que la question de l'utilisateur.

  Répondez en français.  
  Si vous n'arrivez pas à identifier la langue, répondez par défaut en français.
  
  Si la question contient des mots comme "bonjour", "hello", "salam", etc., commencez par :
  "Bonjour! Je suis l'assistant virtuel de l'Université Internationale de Rabat. Comment puis-je vous aider aujourd'hui ? Avez-vous des questions sur nos programmes, les admissions ou peut-être cherchez-vous des informations générales sur l'université ?"
  
  Dans tous les cas, assurez-vous d’inclure dans votre réponse une mention de "l’Université Internationale de Rabat".
  
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

  //   const documentRetrievalChain = RunnableSequence.from([
  //     (input) => input.standalone_question,
  //     retriever,
  //     convertDocsToString,
  //   ]);
  const conversationalRetrievalChain = RunnableSequence.from([
    RunnablePassthrough.assign({
      standalone_question: async (input) => {
        const result = await rephraseQuestionChain.invoke(input);

        return result;
      },
    }),
    RunnablePassthrough.assign({
      context: async (input) => {
        const question = input.standalone_question;
        console.log("✅ Rephrased Question:", question);
        const docs = await retriever.invoke(question);
        const contextString = convertDocsToString(docs);
        return contextString;
      },
    }),
    async (input) => {
      const prompt = await answerGenerationChainPrompt.invoke(input);
      const llm = new ChatOpenAI({ modelName: "gpt-3.5-turbo" });
      const llmResult = await llm.invoke(prompt);
      const parsed = await new StringOutputParser().invoke(llmResult);
      return parsed;
    },
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
