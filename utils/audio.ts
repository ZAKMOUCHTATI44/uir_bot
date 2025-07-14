const { OpenAI } = require("openai");
const axios = require("axios");
const fs = require("fs");

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

export async function handleAudio(mediaUrl: string) {
  if (!mediaUrl) {
    return "";
  }

  console.log("Downloading audio from:", mediaUrl);
  const fileName = "received_audio.ogg";

  try {
    const response = await axios({
      method: "GET",
      url: mediaUrl,
      responseType: "stream",
      headers: {
        Authorization: `Basic ${Buffer.from(
          `${process.env.TWILIO_ACCOUNT_SID}:${process.env.TWILIO_AUTH_TOKEN}`
        ).toString("base64")}`,
      },
    });

    await new Promise((resolve, reject) => {
      const writer = fs.createWriteStream(fileName);
      response.data.pipe(writer);
      writer.on("finish", resolve);
      writer.on("error", reject);
    });

    console.log("Audio saved as", fileName);

    // Transcribe the audio using OpenAI Whisper API
    const transcription = await openai.audio.transcriptions.create({
      file: fs.createReadStream(fileName),
      model: "whisper-1",
    });

    console.log("Transcription:", transcription.text);
    return transcription.text;
  } catch (error) {
    console.error("Error processing audio:", error);
    return "Error processing audio";
  }
}
