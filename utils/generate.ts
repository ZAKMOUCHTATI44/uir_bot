import { getChainWithHistory } from "./history";

export const genrateAnswer = async (userInput: string, phoneNumber: string) => {
  const chainWithHistory: any = await getChainWithHistory();
  const res = await chainWithHistory.invoke(
    {
      input: userInput,
      question: userInput,
    },
    { configurable: { sessionId: phoneNumber } }
  );

  return res;
};
