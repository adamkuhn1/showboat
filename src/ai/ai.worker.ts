import { serveAiTurn, type FromWorker, type Port, type ToWorker } from "./workerProtocol";

// The AI's turn runs here, off the main thread, so the table and controls keep
// animating while it verifies shots. All the logic is in workerProtocol.ts.
serveAiTurn(self as unknown as Port<ToWorker, FromWorker>);
