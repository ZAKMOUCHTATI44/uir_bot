import { twilioInstance } from "./twilio";

export async function sendMessage(sendTo: string, message: string) {
  const client = await twilioInstance();
  client.messages
    .create({
      body: message,
      from: "whatsapp:+212719507879",
      to: sendTo,
    })
    .then(() => {
      console.log("SENDING ...");
    })
    .catch(() => {
      console.log("err");
    });
}
