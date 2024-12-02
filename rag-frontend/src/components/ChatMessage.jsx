// ChatMessage.jsx
import React from 'react';
import ReactMarkdown from 'react-markdown';

const ChatMessage = ({ content, role }) => {
    // Convert string newlines to <br/> tags and preserve markdown
    const formattedContent = content.split('\n').map((line, i) => (
      <React.Fragment key={i}>
        <ReactMarkdown className="inline">{line}</ReactMarkdown>
        <br />
      </React.Fragment>
    ));
  
    // Define styles based on the role
    const messageStyles = {
      user: {
        backgroundColor: '#003585', // Blue for user messages
        color: 'white',
        marginLeft: 'auto', // Align to the right
        maxWidth: '65%', // Limit width
        borderRadius: '8px', // Rounded corners
        padding: '1em', // Padding
        boxShadow: '0 2px 5px rgba(0, 0, 0, 0.2)', // Shadow effect
      },
      assistant: {
        backgroundColor: '#1a5db4', // Gray for assistant messages
        color: 'white',
        marginRight: 'auto', // Align to the left
        maxWidth: '75%', // Limit width
        borderRadius: '8px', // Rounded corners
        padding: '1em', // Padding
        boxShadow: '0 2px 5px rgba(0, 0, 0, 0.2)', // Shadow effect
      },
    };
  
    return (
      <div style={role === 'user' ? messageStyles.user : messageStyles.assistant}>
        {formattedContent}
      </div>
    );
  };
  
  export default ChatMessage;
