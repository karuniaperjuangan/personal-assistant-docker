'use client';
import { useState } from 'react';
import type { NextPage } from 'next';
interface Chat{
  content: string;
  role: string;
}

const base_url = 'http://localhost:8080/chat/completions'

const Chatbot: NextPage = () => {
  const [messages, setMessages] = useState<Chat[]>([]);
  const [input, setInput] = useState<string>('');

  // POST to the API to get the response
  const getResponse = async (input: Chat[]) => {
    const response = await fetch(base_url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        "messages": input,
      }
      )
    });
    const data = await response.json();
    return data.choices[0].message.content;
  }
  const handleSendMessage = async () => {
    let response = await getResponse([...messages,{content: input, role: 'user'}]) as string
    console.log(response)
    if (input.trim()) {
      setMessages((prevMessages) => [...prevMessages, {
      // add the message to the array
      content: input,
      // set the role to user
      role: 'user'
      },
      // get the response from the API
      {
        content: response,
        role: 'assistant'
      }
      ]);
      setInput('');
      console.log(messages)
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-sky-200 p-4 text-black">
      <h1 className="text-2xl font-bold mb-4">Dummy Chatbot</h1>
      <div className="w-full max-w-md bg-white rounded shadow p-4">
        <div className="h-64 overflow-y-scroll mb-4 border-b border-gray-300 w-full">
          {messages.map((message, index) => (
            <div key={index} className={`my-2 text-wrap flex ${index % 2 === 0 ? 'justify-end' : 'justify-start'}`}>
              <p className={`rounded-md p-2 max-w-[75%] break-words ${index % 2 === 0 ? 'text-right bg-gray-200' : 'text-left bg-red-200'}`}>
                {message.content}
              </p>
            </div>
          ))}
        </div>
        <div className="flex items-center">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            className="flex-1 border border-gray-300 p-2 rounded mr-2"
          />
          <button
            onClick={handleSendMessage}
            className="bg-blue-500 text-white px-4 py-2 rounded"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
};

export default Chatbot;
