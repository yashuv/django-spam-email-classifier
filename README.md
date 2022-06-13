# django-spam-email-classifier
This is the '𝒕𝒆𝒙𝒕 𝒃𝒂𝒔𝒆𝒅 𝑬-𝑴𝒂𝒊𝒍 𝑺𝒑𝒂𝒎 𝑪𝒍𝒂𝒔𝒔𝒊𝒇𝒊𝒄𝒂𝒕𝒊𝒐𝒏 𝑷𝒓𝒐𝒋𝒆𝒄𝒕' using a probabilistic machine learning algorithm called 𝑵𝒂ï𝒗𝒆 𝑩𝒂𝒚𝒆𝒔. After developing the working model, it is integrated into a 𝑫𝒋𝒂𝒏𝒈𝒐 application.
<br><br>
In this project I learnt and worked on important steps like:
<pre>
  🔸 𝑬𝑫𝑨 (𝑬𝒙𝒑𝒍𝒐𝒓𝒂𝒕𝒐𝒓𝒚 𝒅𝒂𝒕𝒂 𝒂𝒏𝒂𝒍𝒚𝒔𝒊𝒔)
          - Get the email data
          - Explore and analyze the data
          - Visualize the training data with Word Cloud & Bar Chart
          
  🔸 𝑫𝒂𝒕𝒂 𝑷𝒓𝒆𝒑𝒓𝒐𝒄𝒆𝒔𝒔𝒊𝒏𝒈
          - Text Cleaning Procedures
              ⚬  converting all words in document to lower case
              ⚬  Tokenizing
              ⚬  Removing stop words
              ⚬  Word stemming
              ⚬  Word lemmatization
              ⚬  Removing punctuations
              ⚬  Stripping out HTML tags
          𝐍𝐋𝐓𝐊 library was there to think out of the box!
              
  🔸 𝑭𝒆𝒂𝒕𝒖𝒓𝒆 𝑬𝒙𝒕𝒓𝒂𝒄𝒕𝒊𝒐𝒏
          - CountVectorizer Method
          - Full Matrix Creation
          - Vocabulary Creation
          
  🔸 𝑨𝒍𝒈𝒐𝒓𝒊𝒕𝒉𝒎 𝑰𝒎𝒑𝒍𝒆𝒎𝒆𝒏𝒕𝒂𝒕𝒊𝒐𝒏
          - CountVectorizer + Naïve Bayes Algorithm
          
  🔸 𝑺𝒄𝒐𝒓𝒊𝒏𝒈 & 𝑴𝒆𝒕𝒓𝒊𝒄𝒔 :
          - Accuracy
          - Precision
          - Recall
</pre>
Below is the simple UI of the E-mail classifier project where user enters the message in the text box and the Machine Learning Model predicts and display the result as 'Spam/Ham' when user clicks the 'Predict' button.

<img width="610" alt="ui of spam email classifier project" src="https://user-images.githubusercontent.com/66567559/173371349-112206c6-854e-4e4b-b95b-87d15334b4cf.png">
<br>
Thank You.. HappY Coding :)
