import express from "express";
const app = express();
import bodyParser from "body-parser";
import { sendMessage } from "./utils/message";
import { mainFunction } from "./utils/llm";
import { handleAudio } from "./utils/audio";

app.use(express.urlencoded({ extended: true }));
app.use(express.json());
app.use(bodyParser.json());

app.post("/uir-chat-bot", async (req, res) => {
  const message = req.body;
  if (message.MediaContentType0 === "audio/ogg") {
    const question = await handleAudio(message.MediaUrl0);
    const answer = await mainFunction(question, message.From);
    await sendMessage(message.From, answer);
    res.send({ message: message.Body, answer });
  } else {
    const answer = await mainFunction(message.Body, message.From);
    await sendMessage(message.From, answer);
    res.send({ message: message.Body, answer });
  }
});

const PORT = 7001;
app.listen(PORT, async () => {
  console.log(`App Started in port ${PORT}`);
});
