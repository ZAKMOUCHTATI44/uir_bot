import express from "express";
const app = express();
import bodyParser from "body-parser";
import { getVectoreStore } from "./utils/vector";
import { genrateAnswer } from "./utils/generate";
import { sendMessage } from "./utils/message";
import { mainFunction } from "./utils/llm";

app.use(express.urlencoded({ extended: true }));
app.use(express.json());
app.use(bodyParser.json());
let vectorStore: any;

app.post("/uir-chat-bot", async (req, res) => {
  const message = req.body;
  const answer = await mainFunction(message.Body, message.From);
  await sendMessage(message.From, answer);
  res.send({ message: message.Body, answer });
});

const PORT = 7001;
app.listen(PORT, async () => {
  vectorStore = await getVectoreStore();
  console.log(`App Started in port ${PORT}`);
});
