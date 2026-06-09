import SwiftUI

@main
struct MokuPredictorApp: App {
    var body: some Scene {
        WindowGroup {
            MainView()
                .frame(minWidth: 1120, minHeight: 760)
        }
        .windowStyle(.titleBar)

        Settings {
            SettingsView()
        }
    }
}

