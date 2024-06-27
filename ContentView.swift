import SwiftUI

struct ContentView: View {
    @State private var userInput: String = ""
    @State private var messages: [(String, Bool)] = []

    var body: some View {
        VStack {
            Text("Stock Assistant")
                .font(.largeTitle)
                .fontWeight(.bold)
                .foregroundColor(.black)
                .padding()

            ScrollView {
                ForEach(messages.indices, id: \.self) { index in
                    let message = messages[index]
                    HStack {
                        if message.1 {
                            Spacer()
                            Text(message.0)
                                .padding()
                                .background(Color.blue)
                                .foregroundColor(.white)
                                .cornerRadius(10)
                                .frame(maxWidth: 250, alignment: .trailing)
                        } else {
                            Text(message.0)
                                .padding()
                                .background(Color.gray.opacity(0.2))
                                .foregroundColor(.black)
                                .cornerRadius(10)
                                .frame(maxWidth: 250, alignment: .leading)
                            Spacer()
                        }
                    }
                    .padding(message.1 ? .leading : .trailing, 50)
                    .padding(.vertical, 2)
                }
            }
            .background(Color.white)
            .cornerRadius(10)
            .padding()

            HStack {
                TextField("Message Assistant:", text: $userInput)
                    .textFieldStyle(RoundedBorderTextFieldStyle())
                    .foregroundColor(.black)
                    .padding()

                Button(action: {
                    sendMessage()
                }) {
                    Text("Send")
                        .padding()
                        .background(Color.blue)
                        .foregroundColor(.white)
                        .cornerRadius(10)
                }
                .padding()
            }
            .padding()
        }
        .background(Color.white)
    }

    func sendMessage() {
        let trimmedMessage = userInput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedMessage.isEmpty else { return }
        
        messages.append((trimmedMessage, true))
        userInput = ""

        guard let url = URL(string: "https://api.openai.com/v1/completions") else {
            print("Invalid URL")
            return
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        request.addValue("Bearer APIKEYHERE", forHTTPHeaderField: "Authorization")
        
        let parameters: [String: Any] = [
            "model": "gpt-4",
            "prompt": trimmedMessage,
            "max_tokens": 200,
            "temperature": 1
        ]
        
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: parameters, options: [])
        } catch let error {
            print("Failed to serialize JSON: \(error.localizedDescription)")
            return
        }
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            if let error = error {
                print("Error making API request: \(error.localizedDescription)")
                return
            }
            
            guard let data = data else {
                print("No data received")
                return
            }
            
            do {
                if let jsonResponse = try JSONSerialization.jsonObject(with: data, options: []) as? [String: Any],
                   let choices = jsonResponse["choices"] as? [[String: Any]],
                   let text = choices.first?["text"] as? String {
                    DispatchQueue.main.async {
                        let aiResponse = text.trimmingCharacters(in: .whitespacesAndNewlines)
                        messages.append((aiResponse, false))
                    }
                } else {
                    print("Invalid JSON response")
                }
            } catch let error {
                print("Failed to parse JSON: \(error.localizedDescription)")
            }
        }.resume()
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
