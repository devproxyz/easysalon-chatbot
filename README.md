# EasySalon Chatbot 💄✨

An AI-powered customer service assistant for beauty salons, built with OpenAI GPT-4 and modern web technologies.

## 🎯 Features

### ✅ Core Functionality (Stories 1-9 Implemented)
- **Appointment Management**: Check availability, book appointments, retrieve booking information
- **Service Discovery**: Browse services, view pricing, get detailed information
- **Location Services**: Find nearby salons, compare locations and ratings
- **Beauty Consultation**: Get personalized beauty advice and recommendations
- **Semantic Search**: Intelligent search across salon services and information
- **Comprehensive Info**: Access salon hours, policies, staff information, and amenities

### 🚀 Technical Features
- **Natural Language Processing**: Powered by OpenAI GPT-4o-mini
- **Function Calling**: Advanced AI tool integration for specific actions
- **Vector Database**: Qdrant for semantic search and recommendations
- **Real-time Chat**: WebSocket-based communication with Flask-SocketIO
- **Responsive Design**: Bootstrap 5.3.2 with mobile-first approach
- **Modern UI**: Clean, professional design with accessibility features

## 🏗️ Architecture

```
EasySalon-Chatbot/
├── app.py                 # Flask web server
├── requirements.txt       # Dependencies
├── templates/
│   └── index.html        # Frontend interface
├── src/
│   ├── chatbot.py        # Main chatbot logic
│   ├── availability_checker.py    # Story 1: Availability
│   ├── booking_manager.py         # Story 2: Booking
│   ├── booking_retriever.py       # Story 3: Booking retrieval
│   ├── service_browser.py         # Story 4: Services
│   ├── salon_finder.py           # Story 5: Salon search
│   ├── beauty_consultant.py      # Story 6: Consultation
│   ├── semantic_search.py        # Story 8: Search
│   ├── salon_info_manager.py     # Story 9: Salon info
│   └── [other modules...]
├── tests/                # Test suite
├── Data/
│   └── locations.json    # Salon location data
└── docs/                 # Documentation
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Virtual environment (recommended)

### Installation
```bash
# Clone the repository
git clone [repository-url]
cd easysalon-chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

### Access the Application
- Open your browser to `http://localhost:5001`
- Start chatting with the EasySalon assistant!

## 💬 Usage Examples

### Sample Conversations
```
User: "I need a haircut appointment for tomorrow morning"
Bot: "I'd be happy to help you book a haircut! Let me check availability for tomorrow morning..."

User: "What services do you offer?"
Bot: "We offer a wide range of beauty services including haircuts, coloring, styling, manicures, pedicures, facials, and more..."

User: "Find salons near downtown"
Bot: "I found several salons in the downtown area. Here are the top options with ratings and contact information..."

User: "What hairstyle would suit my round face?"
Bot: "For a round face shape, I'd recommend layered cuts that add height and volume at the crown..."
```

## 🧪 Testing

### Run Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific module tests
python -m pytest tests/test_chatbot.py -v

# Run simple functionality check
python tests/test_simple.py
```

### Test Coverage
- ✅ Unit tests for all modules
- ✅ Integration tests for complete workflows
- ✅ Mock API testing
- ✅ Error handling validation

## 📊 Implementation Status

| Story | Feature | Status | Module |
|-------|---------|--------|---------|
| 1 | Check Appointment Availability | ✅ Complete | `availability_checker.py` |
| 2 | Book Appointment | ✅ Complete | `booking_manager.py` |
| 3 | Retrieve Booking Information | ✅ Complete | `booking_retriever.py` |
| 4 | Browse Services and Pricing | ✅ Complete | `service_browser.py` |
| 5 | Find Nearby Salons | ✅ Complete | `salon_finder.py` |
| 6 | Beauty Consultation | ✅ Complete | `beauty_consultant.py` |
| 7 | User Interface Enhancements | ✅ Complete | `templates/index.html` |
| 8 | Semantic Search | ✅ Complete | `semantic_search.py` |
| 9 | Comprehensive Salon Information | ✅ Complete | `salon_info_manager.py` |

**Total: 9/9 Stories Complete** 🎉

## 🛠️ Technology Stack

- **Backend**: Python 3.9, Flask, Flask-SocketIO
- **AI/ML**: OpenAI GPT-4o-mini, LangChain
- **Database**: Qdrant Vector Database
- **Frontend**: HTML5, Bootstrap 5.3.2, JavaScript
- **Testing**: pytest, unittest.mock
- **Other**: NLTK, NumPy, Requests

## 📝 Configuration

### Environment Variables
Configure the following in `src/global_vars.py`:
- `AZURE_OPENAI_API_KEY`: Your OpenAI API key
- `QDRANT_HOST`: Qdrant database host
- `QDRANT_API_KEY`: Qdrant API key
- `EASYSALON_API_KEY`: EasySalon API key (for production)

### API Integration
Currently using mock APIs for development. Replace with real EasySalon APIs by:
1. Updating API endpoints in each module
2. Implementing proper authentication
3. Handling real API response formats

## 🔧 Development

### Adding New Features
1. Create new module in `src/`
2. Implement required functions
3. Add to `chatbot.py` tool definitions
4. Update prompts and function calls
5. Add comprehensive tests

### Code Quality
- Type hints throughout the codebase
- Comprehensive error handling
- Structured logging
- Clean, modular architecture
- Full test coverage

## 🚀 Production Deployment

### Requirements
- WSGI server (e.g., Gunicorn)
- Reverse proxy (e.g., Nginx)
- SSL certificate
- Production database
- Environment-specific configuration

### Security Considerations
- API key management
- Input validation
- Rate limiting
- CORS configuration
- HTTPS enforcement

## 📖 Documentation

- **Implementation Guide**: `IMPLEMENTATION.md`
- **User Stories**: `docs/stories/`
- **Feature Specifications**: `docs/feature/`
- **API Documentation**: In respective module docstrings

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Implement changes with tests
4. Submit pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues or questions:
1. Check the documentation
2. Review test examples
3. Open an issue on GitHub
4. Contact the development team

---

**Built with ❤️ for the beauty industry**
