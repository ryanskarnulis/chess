import { Board } from './Board'
import './App.css'

const START_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR'

function App() {
  return (
    <main className="app">
      <h1>Chess</h1>
      <div className="board-wrap">
        <Board fen={START_FEN} />
      </div>
    </main>
  )
}

export default App
